/**
 * Neural Learning Web - Backend Server
 *
 * Express.js server for personal ML learning companion
 * Single-user local application with SQLite database
 *
 * Features:
 * - Learning progress tracking
 * - Content management and delivery
 * - Quest/challenge system
 * - Vault unlock mechanics
 * - Code execution result storage
 * - Automated database backups
 * - Development-optimized CORS and logging
 *
 * Author: Neural Explorer
 * Port: 3001 (Frontend on 3000)
 */

const express = require('express')
const cors = require('cors')
const helmet = require('helmet')
const compression = require('compression')
const morgan = require('morgan')
const path = require('path')
const fs = require('fs').promises
const chalk = require('chalk')

// Database and utilities
const DatabaseManager = require('./config/db')

// Initialize Express app
const app = express()
const PORT = process.env.PORT || 3001
const NODE_ENV = process.env.NODE_ENV || 'development'
const isDevelopment = NODE_ENV === 'development'

// Initialize database
const db = new DatabaseManager()

// Security middleware (relaxed for local single-user setup)
app.use(
  helmet({
    contentSecurityPolicy: false, // Disabled for local development
    crossOriginEmbedderPolicy: false // Allow iframe embedding for Monaco/Pyodide
  })
)

// Compression middleware
app.use(compression())

// CORS configuration for local development
app.use(
  cors({
    origin: [
      'http://localhost:3000',
      'http://127.0.0.1:3000',
      'http://0.0.0.0:3000',
      'http://[::1]:3000'
    ],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
  })
)

// Body parsing middleware
app.use(express.json({ limit: '10mb' })) // Large limit for code submissions
app.use(express.urlencoded({ extended: true, limit: '10mb' }))

// Request logging
if (isDevelopment) {
  app.use(morgan('dev'))
} else {
  app.use(morgan('combined'))
}

// Request timing middleware
app.use((req, res, next) => {
  req.startTime = Date.now()
  res.on('finish', () => {
    const duration = Date.now() - req.startTime
    if (duration > 1000) {
      // Log slow requests
      console.log(
        chalk.yellow(
          `⚠️  Slow request: ${req.method} ${req.path} took ${duration}ms`
        )
      )
    }
  })
  next()
})

// Health check endpoint (no authentication needed)
app.get('/health', async (req, res) => {
  try {
    const dbStats = await db.getStats()
    const healthData = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      database: {
        connected: dbStats.isConnected,
        tables: dbStats.tables,
        dbSize: dbStats.dbSizeMB ? `${dbStats.dbSizeMB}MB` : 'Unknown'
      },
      environment: NODE_ENV,
      version: process.env.npm_package_version || '1.0.0'
    }

    res.json(healthData)
  } catch (error) {
    console.error('Health check failed:', error)
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    })
  }
})

// API Routes

// User Profile & Progress Routes
app.get('/api/profile', async (req, res) => {
  try {
    const profile = await db.get(`
      SELECT * FROM user_profile 
      WHERE id = 1
    `)

    if (!profile) {
      // Create default profile for single user
      await db.run(`
        INSERT INTO user_profile (id, username) 
        VALUES (1, 'Neural Explorer')
      `)
      return res.json({
        id: 1,
        username: 'Neural Explorer',
        created_at: new Date().toISOString(),
        current_phase: 1,
        current_week: 1,
        total_study_minutes: 0,
        current_streak_days: 0
      })
    }

    res.json(profile)
  } catch (error) {
    console.error('Error fetching profile:', error)
    res.status(500).json({ error: 'Failed to fetch user profile' })
  }
})

app.put('/api/profile', async (req, res) => {
  try {
    const { username, timezone, preferred_session_length, daily_goal_minutes } =
      req.body

    await db.run(
      `
      UPDATE user_profile 
      SET username = ?, timezone = ?, preferred_session_length = ?, 
          daily_goal_minutes = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = 1
    `,
      [username, timezone, preferred_session_length, daily_goal_minutes]
    )

    const updatedProfile = await db.get(
      'SELECT * FROM user_profile WHERE id = 1'
    )
    res.json(updatedProfile)
  } catch (error) {
    console.error('Error updating profile:', error)
    res.status(500).json({ error: 'Failed to update profile' })
  }
})

// Learning Progress Routes
app.get('/api/learning/progress', async (req, res) => {
  try {
    // Get overall progress
    const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

    // Get phase progress
    const phaseProgress = await db.all(`
      SELECT 
        phase_number,
        COUNT(*) as total_weeks,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_weeks,
        SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_weeks
      FROM learning_progress 
      GROUP BY phase_number
      ORDER BY phase_number
    `)

    // Get recent activity
    const recentActivity = await db.all(`
      SELECT * FROM daily_sessions 
      ORDER BY session_date DESC 
      LIMIT 7
    `)

    // Calculate streak
    const today = new Date().toISOString().split('T')[0]
    const yesterdayActivity = await db.get(`
      SELECT COUNT(*) as count FROM daily_sessions 
      WHERE session_date = date('now', '-1 day')
    `)

    res.json({
      profile,
      phases: phaseProgress,
      recentActivity,
      streakActive: yesterdayActivity?.count > 0,
      currentDate: today
    })
  } catch (error) {
    console.error('Error fetching learning progress:', error)
    res.status(500).json({ error: 'Failed to fetch learning progress' })
  }
})

app.post('/api/learning/session', async (req, res) => {
  try {
    const { session_type, duration_minutes, focus_score, phase, week } =
      req.body

    // Record session
    await db.run(
      `
      INSERT INTO daily_sessions (session_date, session_type, duration_minutes, focus_score)
      VALUES (date('now'), ?, ?, ?)
    `,
      [session_type, duration_minutes, focus_score]
    )

    // Update progress if provided
    if (phase && week) {
      await db.run(
        `
        INSERT OR REPLACE INTO learning_progress 
        (phase_number, week_number, status, progress_percentage, last_accessed, updated_at)
        VALUES (?, ?, 'in_progress', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
      `,
        [phase, week, Math.min(100, (duration_minutes / 60) * 25)]
      ) // Rough progress calculation
    }

    // Update user profile stats
    await db.run(
      `
      UPDATE user_profile 
      SET total_study_minutes = total_study_minutes + ?,
          last_activity = CURRENT_TIMESTAMP,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = 1
    `,
      [duration_minutes]
    )

    res.json({ success: true, message: 'Session recorded successfully' })
  } catch (error) {
    console.error('Error recording session:', error)
    res.status(500).json({ error: 'Failed to record session' })
  }
})

// Content Routes
app.get('/api/content/phases', async (req, res) => {
  try {
    // Get all phases with progress
    const phases = await db.all(`
      SELECT 
        p.phase_number,
        p.week_number,
        p.status,
        p.progress_percentage,
        p.last_accessed,
        CASE 
          WHEN p.status = 'completed' THEN 'completed'
          WHEN p.status = 'in_progress' THEN 'active' 
          WHEN p.phase_number = 1 AND p.week_number = 1 THEN 'available'
          WHEN EXISTS(
            SELECT 1 FROM learning_progress p2 
            WHERE p2.phase_number = p.phase_number - 1 
            AND p2.status = 'completed'
          ) THEN 'available'
          ELSE 'locked'
        END as availability
      FROM learning_progress p
      ORDER BY p.phase_number, p.week_number
    `)

    // Group by phases
    const phaseMap = {}
    phases.forEach(item => {
      if (!phaseMap[item.phase_number]) {
        phaseMap[item.phase_number] = {
          phase: item.phase_number,
          weeks: [],
          totalWeeks: 0,
          completedWeeks: 0
        }
      }

      phaseMap[item.phase_number].weeks.push({
        week: item.week_number,
        status: item.status,
        progress: item.progress_percentage,
        lastAccessed: item.last_accessed,
        availability: item.availability
      })

      phaseMap[item.phase_number].totalWeeks++
      if (item.status === 'completed') {
        phaseMap[item.phase_number].completedWeeks++
      }
    })

    res.json(Object.values(phaseMap))
  } catch (error) {
    console.error('Error fetching content phases:', error)
    res.status(500).json({ error: 'Failed to fetch content phases' })
  }
})

app.get('/api/content/phase/:phase/week/:week', async (req, res) => {
  try {
    const { phase, week } = req.params

    // Get week progress
    const progress = await db.get(
      `
      SELECT * FROM learning_progress 
      WHERE phase_number = ? AND week_number = ?
    `,
      [phase, week]
    )

    // For now, return mock content structure
    // In a complete implementation, this would read from content files
    const weekContent = {
      phase: parseInt(phase),
      week: parseInt(week),
      title: `Phase ${phase} - Week ${week}`,
      description: `Learning content for Phase ${phase}, Week ${week}`,
      progress: progress?.progress_percentage || 0,
      status: progress?.status || 'not_started',
      lessons: [
        {
          id: `${phase}-${week}-1`,
          title: 'Theoretical Foundation',
          type: 'theory',
          duration: 45,
          completed: progress?.status === 'completed'
        },
        {
          id: `${phase}-${week}-2`,
          title: 'Practical Implementation',
          type: 'coding',
          duration: 60,
          completed: progress?.status === 'completed'
        },
        {
          id: `${phase}-${week}-3`,
          title: 'Visual Project',
          type: 'visualization',
          duration: 30,
          completed: progress?.status === 'completed'
        }
      ],
      resources: [
        { type: 'pdf', title: 'Reading Material', url: '#' },
        { type: 'video', title: 'Tutorial Video', url: '#' },
        { type: 'code', title: 'Code Examples', url: '#' }
      ]
    }

    // Update last accessed
    await db.run(
      `
      INSERT OR REPLACE INTO learning_progress 
      (phase_number, week_number, status, progress_percentage, last_accessed, updated_at)
      VALUES (?, ?, COALESCE(?, 'not_started'), COALESCE(?, 0), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    `,
      [phase, week, progress?.status, progress?.progress_percentage]
    )

    res.json(weekContent)
  } catch (error) {
    console.error('Error fetching week content:', error)
    res.status(500).json({ error: 'Failed to fetch week content' })
  }
})

// Quest Routes
app.get('/api/quests', async (req, res) => {
  try {
    const quests = await db.all(`
      SELECT * FROM quests 
      ORDER BY created_at DESC
    `)

    res.json(quests || [])
  } catch (error) {
    console.error('Error fetching quests:', error)
    res.status(500).json({ error: 'Failed to fetch quests' })
  }
})

app.post('/api/quests/:questId/complete', async (req, res) => {
  try {
    const { questId } = req.params
    const { code_submission, result } = req.body

    await db.run(
      `
      UPDATE quests 
      SET status = 'completed', 
          completion_date = CURRENT_TIMESTAMP,
          code_submission = ?,
          result_data = ?,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
      [code_submission, JSON.stringify(result), questId]
    )

    // Award experience points
    await db.run(`
      UPDATE user_profile 
      SET experience_points = experience_points + 100,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = 1
    `)

    res.json({ success: true, message: 'Quest completed successfully' })
  } catch (error) {
    console.error('Error completing quest:', error)
    res.status(500).json({ error: 'Failed to complete quest' })
  }
})

// Vault Routes
app.get('/api/vault', async (req, res) => {
  try {
    const vaultItems = await db.all(`
      SELECT * FROM vault_items 
      ORDER BY unlock_requirement_type, created_at
    `)

    res.json(vaultItems || [])
  } catch (error) {
    console.error('Error fetching vault items:', error)
    res.status(500).json({ error: 'Failed to fetch vault items' })
  }
})

app.post('/api/vault/:itemId/unlock', async (req, res) => {
  try {
    const { itemId } = req.params

    await db.run(
      `
      UPDATE vault_items 
      SET unlocked = 1, 
          unlocked_at = CURRENT_TIMESTAMP,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `,
      [itemId]
    )

    res.json({ success: true, message: 'Vault item unlocked successfully' })
  } catch (error) {
    console.error('Error unlocking vault item:', error)
    res.status(500).json({ error: 'Failed to unlock vault item' })
  }
})

// Code Execution Results Storage
app.post('/api/code/save', async (req, res) => {
  try {
    const { lesson_id, code, result, execution_time } = req.body

    await db.run(
      `
      INSERT OR REPLACE INTO code_submissions 
      (lesson_id, code, result, execution_time, created_at)
      VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    `,
      [lesson_id, code, JSON.stringify(result), execution_time]
    )

    res.json({ success: true, message: 'Code execution saved' })
  } catch (error) {
    console.error('Error saving code execution:', error)
    res.status(500).json({ error: 'Failed to save code execution' })
  }
})

// Database Management Routes
app.get('/api/admin/database/stats', async (req, res) => {
  try {
    const stats = await db.getStats()
    res.json(stats)
  } catch (error) {
    console.error('Error fetching database stats:', error)
    res.status(500).json({ error: 'Failed to fetch database stats' })
  }
})

app.post('/api/admin/database/backup', async (req, res) => {
  try {
    const backupPath = await db.backup()
    res.json({
      success: true,
      message: 'Database backup created successfully',
      backupPath
    })
  } catch (error) {
    console.error('Error creating backup:', error)
    res.status(500).json({ error: 'Failed to create database backup' })
  }
})

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(chalk.red('❌ Server Error:'), err)

  // Don't leak error details in production
  const errorMessage = isDevelopment ? err.message : 'Internal server error'
  const errorStack = isDevelopment ? err.stack : undefined

  res.status(err.status || 500).json({
    error: errorMessage,
    stack: errorStack,
    timestamp: new Date().toISOString(),
    path: req.path
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: new Date().toISOString()
  })
})

// Initialize database and start server
async function startServer () {
  try {
    console.log(chalk.blue('🔗 Connecting to database...'))
    await db.connect()

    console.log(chalk.blue('🔨 Initializing database schema...'))
    await db.initializeSchema()

    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(
        chalk.green('\n🚀 Neural Learning Web Backend Server Started!')
      )
      console.log(
        chalk.white('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
      )
      console.log(chalk.cyan(`   🌐 Server: http://localhost:${PORT}`))
      console.log(chalk.cyan(`   🔗 Health: http://localhost:${PORT}/health`))
      console.log(chalk.cyan(`   📊 API: http://localhost:${PORT}/api`))
      console.log(
        chalk.white('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
      )
      console.log(chalk.gray(`   Environment: ${NODE_ENV}`))
      console.log(chalk.gray(`   Database: SQLite (${db.path})`))
      console.log(chalk.gray('   Single-user mode: No authentication required'))
      console.log(chalk.gray('   CORS: Enabled for frontend development'))
      console.log(chalk.gray('   Ready for frontend connection on port 3000\n'))
    })

    // Graceful shutdown
    process.on('SIGTERM', () => gracefulShutdown(server))
    process.on('SIGINT', () => gracefulShutdown(server))

    return server
  } catch (error) {
    console.error(chalk.red('❌ Failed to start server:'), error)
    process.exit(1)
  }
}

async function gracefulShutdown (server) {
  console.log(chalk.yellow('\n🛑 Graceful shutdown initiated...'))

  server.close(async () => {
    console.log(chalk.blue('📴 HTTP server closed'))

    try {
      await db.close()
      console.log(chalk.blue('🗄️  Database connection closed'))
    } catch (error) {
      console.error(chalk.red('❌ Error closing database:'), error)
    }

    console.log(chalk.green('✅ Graceful shutdown complete'))
    process.exit(0)
  })

  // Force close after 10 seconds
  setTimeout(() => {
    console.log(chalk.red('⚠️  Forced shutdown after timeout'))
    process.exit(1)
  }, 10000)
}

// Start the server
if (require.main === module) {
  startServer()
}

module.exports = { app, startServer }
