/**
 * Neural Odyssey Backend Server - FIXED VERSION
 *
 * Now properly reads content files instead of serving mock data
 * Handles all 4 session types: math, coding, visual_projects, real_applications
 * Reads content.json, lesson.md, project.md, visualization.py files
 *
 * Author: Neural Explorer
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const path = require('path');
const fs = require('fs').promises;
const chalk = require('chalk');
const matter = require('gray-matter'); // For parsing markdown frontmatter

// Load environment variables
require('dotenv').config();

// Import database configuration
const db = require('./config/db');

// Initialize Express app
const app = express();

// Configuration
const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';
const isDevelopment = NODE_ENV === 'development';

// Trust proxy for accurate IP addresses
app.set('trust proxy', 1);

// Security Middleware
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
        scriptSrc: ["'self'", "'unsafe-eval'", 'https://cdnjs.cloudflare.com'],
        imgSrc: ["'self'", 'data:', 'https:'],
        connectSrc: ["'self'", 'https://cdnjs.cloudflare.com'],
        fontSrc: ["'self'", 'https://fonts.gstatic.com'],
        workerSrc: ["'self'", 'blob:'],
      },
    },
    crossOriginEmbedderPolicy: false,
  })
);

// CORS Configuration for local development
const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests with no origin (mobile apps, Postman, etc.)
    if (!origin) return callback(null, true);

    // Allow localhost development
    if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
      return callback(null, true);
    }

    // Allow any origin in development
    if (isDevelopment) {
      return callback(null, true);
    }

    callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Compression middleware
app.use(compression());

// Logging middleware
if (isDevelopment) {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// Static file serving for content files (with CORS headers)
app.use(
  '/content',
  express.static(path.join(__dirname, '..', 'content'), {
    setHeaders: (res, path) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    },
  })
);

// ==========================================
// CONTENT READING UTILITIES
// ==========================================

/**
 * Get phase name from phase number
 */
function getPhaseNames () {
  return {
    1: 'foundations',
    2: 'core-ml',
    3: 'advanced',
    4: 'mastery',
  };
}

/**
 * Get week slug from directory structure
 */
async function getWeekSlug (phase, week) {
  const phaseDir = path.join(
    __dirname,
    '..',
    'content',
    'phases',
    `phase-${phase}-${getPhaseNames()[phase]}`
  );

  try {
    const dirs = await fs.readdir(phaseDir);
    const weekDir = dirs.find(dir => dir.startsWith(`week-${week}-`));
    return weekDir ? weekDir.replace(`week-${week}-`, '') : 'unknown';
  } catch (error) {
    console.error(`Error reading phase directory: ${error.message}`);
    return 'unknown';
  }
}

/**
 * Read and parse content.json file
 */
async function readContentJson (contentDir) {
  try {
    const contentJsonPath = path.join(contentDir, 'content.json');
    const contentExists = await fs
      .access(contentJsonPath)
      .then(() => true)
      .catch(() => false);

    if (contentExists) {
      const contentData = await fs.readFile(contentJsonPath, 'utf8');
      return JSON.parse(contentData);
    }
    return null;
  } catch (error) {
    console.error(`Error reading content.json: ${error.message}`);
    return null;
  }
}

/**
 * Read and parse lesson.md file
 */
async function readLessonMd (contentDir) {
  try {
    const lessonPath = path.join(contentDir, 'lesson.md');
    const lessonExists = await fs
      .access(lessonPath)
      .then(() => true)
      .catch(() => false);

    if (lessonExists) {
      const lessonContent = await fs.readFile(lessonPath, 'utf8');
      const parsed = matter(lessonContent);
      return {
        frontmatter: parsed.data,
        content: parsed.content,
      };
    }
    return null;
  } catch (error) {
    console.error(`Error reading lesson.md: ${error.message}`);
    return null;
  }
}

/**
 * Read project.md file
 */
async function readProjectMd (contentDir) {
  try {
    const projectPath = path.join(contentDir, 'project.md');
    const projectExists = await fs
      .access(projectPath)
      .then(() => true)
      .catch(() => false);

    if (projectExists) {
      const projectContent = await fs.readFile(projectPath, 'utf8');
      const parsed = matter(projectContent);
      return {
        frontmatter: parsed.data,
        content: parsed.content,
      };
    }
    return null;
  } catch (error) {
    console.error(`Error reading project.md: ${error.message}`);
    return null;
  }
}

/**
 * Read visualization.py file
 */
async function readVisualizationPy (contentDir) {
  try {
    const vizPath = path.join(contentDir, 'visualization.py');
    const vizExists = await fs
      .access(vizPath)
      .then(() => true)
      .catch(() => false);

    if (vizExists) {
      const vizContent = await fs.readFile(vizPath, 'utf8');
      return {
        language: 'python',
        content: vizContent,
        hasImplementation: !vizContent.includes('TODO') && !vizContent.includes('pass'),
        functions: extractPythonFunctions(vizContent),
      };
    }
    return null;
  } catch (error) {
    console.error(`Error reading visualization.py: ${error.message}`);
    return null;
  }
}

/**
 * Read exercises.py file
 */
async function readExercisesPy (contentDir) {
  try {
    const exercisesPath = path.join(contentDir, 'exercises.py');
    const exercisesExists = await fs
      .access(exercisesPath)
      .then(() => true)
      .catch(() => false);

    if (exercisesExists) {
      const exercisesContent = await fs.readFile(exercisesPath, 'utf8');
      return {
        language: 'python',
        content: exercisesContent,
        hasImplementation: !exercisesContent.includes('TODO'),
        functions: extractPythonFunctions(exercisesContent),
      };
    }
    return null;
  } catch (error) {
    console.error(`Error reading exercises.py: ${error.message}`);
    return null;
  }
}

/**
 * Read resources.json file
 */
async function readResourcesJson (contentDir) {
  try {
    const resourcesPath = path.join(contentDir, 'resources.json');
    const resourcesExists = await fs
      .access(resourcesPath)
      .then(() => true)
      .catch(() => false);

    if (resourcesExists) {
      const resourcesContent = await fs.readFile(resourcesPath, 'utf8');
      return JSON.parse(resourcesContent);
    }
    return null;
  } catch (error) {
    console.error(`Error reading resources.json: ${error.message}`);
    return null;
  }
}

/**
 * Extract Python function names from code
 */
function extractPythonFunctions (code) {
  const functionPattern = /def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g;
  const functions = [];
  let match;

  while ((match = functionPattern.exec(code)) !== null) {
    functions.push(match[1]);
  }

  return functions;
}

/**
 * Main function to read complete week content
 */
async function readWeekContent (contentDir, phase, week, progress) {
  const weekContent = {
    phase: phase,
    week: week,
    title: `Phase ${phase} - Week ${week}`,
    description: '',
    progress: progress?.progress_percentage || 0,
    status: progress?.status || 'not_started',
    lastAccessed: progress?.last_accessed || null,
    sessions: {},
    vault_rewards: [],
    resources: [],
    file_structure: {
      content_json: false,
      lesson_md: false,
      project_md: false,
      visualization_py: false,
      exercises_py: false,
      resources_json: false,
    },
  };

  try {
    // Read content.json (session structure)
    const contentJson = await readContentJson(contentDir);
    if (contentJson) {
      weekContent.file_structure.content_json = true;
      weekContent.title = contentJson.week_metadata?.title || weekContent.title;
      weekContent.description = contentJson.week_metadata?.description || weekContent.description;
      weekContent.sessions = contentJson.daily_sessions || {};
      weekContent.vault_rewards = contentJson.vault_rewards || [];
      weekContent.learning_objectives = contentJson.week_metadata?.learning_objectives || [];
      weekContent.estimated_time = contentJson.week_metadata?.estimated_total_time || '6-8 hours';
      weekContent.difficulty = contentJson.week_metadata?.difficulty_level || 'foundational';
    }

    // Read lesson.md (math session content)
    const lessonMd = await readLessonMd(contentDir);
    if (lessonMd) {
      weekContent.file_structure.lesson_md = true;
      if (weekContent.sessions.math) {
        weekContent.sessions.math.content = lessonMd.content;
        weekContent.sessions.math.frontmatter = lessonMd.frontmatter;
      }
    }

    // Read project.md (visual_projects and real_applications content)
    const projectMd = await readProjectMd(contentDir);
    if (projectMd) {
      weekContent.file_structure.project_md = true;
      if (weekContent.sessions.visual_projects) {
        weekContent.sessions.visual_projects.project_content = projectMd.content;
        weekContent.sessions.visual_projects.project_frontmatter = projectMd.frontmatter;
      }
      if (weekContent.sessions.real_applications) {
        weekContent.sessions.real_applications.project_content = projectMd.content;
        weekContent.sessions.real_applications.project_frontmatter = projectMd.frontmatter;
      }
    }

    // Read visualization.py (visual_projects executable content)
    const visualizationPy = await readVisualizationPy(contentDir);
    if (visualizationPy) {
      weekContent.file_structure.visualization_py = true;
      if (weekContent.sessions.visual_projects) {
        weekContent.sessions.visual_projects.code = visualizationPy;
      }
    }

    // Read exercises.py (coding session content)
    const exercisesPy = await readExercisesPy(contentDir);
    if (exercisesPy) {
      weekContent.file_structure.exercises_py = true;
      if (weekContent.sessions.coding) {
        weekContent.sessions.coding.code = exercisesPy;
      }
    }

    // Read resources.json
    const resourcesJson = await readResourcesJson(contentDir);
    if (resourcesJson) {
      weekContent.file_structure.resources_json = true;
      weekContent.resources = resourcesJson;
    }

    // Generate session availability based on file structure
    weekContent.session_availability = {
      math: weekContent.file_structure.lesson_md,
      coding: weekContent.file_structure.exercises_py,
      visual_projects:
        weekContent.file_structure.visualization_py && weekContent.file_structure.project_md,
      real_applications: weekContent.file_structure.project_md,
    };
  } catch (error) {
    console.error(`Error reading content from ${contentDir}:`, error.message);
  }

  return weekContent;
}

// ==========================================
// API ROUTES
// ==========================================

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    environment: NODE_ENV,
    version: '1.0.0',
  });
});

// Basic API info
app.get('/api', (req, res) => {
  res.json({
    name: 'Neural Odyssey API',
    version: '1.0.0',
    environment: NODE_ENV,
    endpoints: {
      health: '/health',
      content: '/api/content/*',
      progress: '/api/progress/*',
      vault: '/api/vault/*',
      sessions: '/api/sessions/*',
    },
  });
});

// ==========================================
// CONTENT ROUTES - COMPLETELY REWRITTEN
// ==========================================

// Get all phases overview
app.get('/api/content/phases', async (req, res) => {
  try {
    // Get progress data for all phases
    const progressData = await db.all(`
            SELECT 
                phase_number,
                week_number,
                status,
                progress_percentage,
                last_accessed
            FROM learning_progress
            ORDER BY phase_number, week_number
        `);

    // Group progress by phases
    const phaseMap = {};
    progressData.forEach(item => {
      if (!phaseMap[item.phase_number]) {
        phaseMap[item.phase_number] = {
          phase: item.phase_number,
          weeks: [],
          totalWeeks: 0,
          completedWeeks: 0,
          totalProgress: 0,
        };
      }

      phaseMap[item.phase_number].weeks.push({
        week: item.week_number,
        status: item.status,
        progress: item.progress_percentage,
        lastAccessed: item.last_accessed,
      });

      phaseMap[item.phase_number].totalWeeks++;
      if (item.status === 'completed') {
        phaseMap[item.phase_number].completedWeeks++;
      }
      phaseMap[item.phase_number].totalProgress += item.progress_percentage || 0;
    });

    // Calculate average progress for each phase
    Object.values(phaseMap).forEach(phase => {
      phase.averageProgress = phase.totalWeeks > 0 ? phase.totalProgress / phase.totalWeeks : 0;
    });

    res.json({
      success: true,
      data: Object.values(phaseMap),
    });
  } catch (error) {
    console.error('Error fetching phases:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch phases',
      details: error.message,
    });
  }
});

// Get specific week content - COMPLETELY REWRITTEN
app.get('/api/content/phase/:phase/week/:week', async (req, res) => {
  try {
    const { phase, week } = req.params;
    const phaseNumber = parseInt(phase);
    const weekNumber = parseInt(week);

    // Validate parameters
    if (
      isNaN(phaseNumber) ||
      isNaN(weekNumber) ||
      phaseNumber < 1 ||
      phaseNumber > 4 ||
      weekNumber < 1
    ) {
      return res.status(400).json({
        success: false,
        error: 'Invalid phase or week number',
      });
    }

    // Get week progress from database
    const progress = await db.get(
      `SELECT * FROM learning_progress WHERE phase_number = ? AND week_number = ?`,
      [phaseNumber, weekNumber]
    );

    // Construct content directory path
    const phaseNames = getPhaseNames();
    const weekSlug = await getWeekSlug(phaseNumber, weekNumber);
    const contentDir = path.join(
      __dirname,
      '..',
      'content',
      'phases',
      `phase-${phaseNumber}-${phaseNames[phaseNumber]}`,
      `week-${weekNumber}-${weekSlug}`
    );

    // Read actual content files
    const weekContent = await readWeekContent(contentDir, phaseNumber, weekNumber, progress);

    // Update last accessed in database
    await db.run(
      `INSERT OR REPLACE INTO learning_progress 
             (phase_number, week_number, status, progress_percentage, last_accessed, updated_at)
             VALUES (?, ?, COALESCE(?, 'not_started'), COALESCE(?, 0), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [phaseNumber, weekNumber, progress?.status, progress?.progress_percentage]
    );

    res.json({
      success: true,
      data: weekContent,
    });
  } catch (error) {
    console.error('Error fetching week content:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch week content',
      details: error.message,
    });
  }
});

// Get specific session content
app.get('/api/content/phase/:phase/week/:week/session/:sessionType', async (req, res) => {
  try {
    const { phase, week, sessionType } = req.params;
    const phaseNumber = parseInt(phase);
    const weekNumber = parseInt(week);

    // Validate session type
    const validSessionTypes = ['math', 'coding', 'visual_projects', 'real_applications'];
    if (!validSessionTypes.includes(sessionType)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid session type',
      });
    }

    // Get week content
    const phaseNames = getPhaseNames();
    const weekSlug = await getWeekSlug(phaseNumber, weekNumber);
    const contentDir = path.join(
      __dirname,
      '..',
      'content',
      'phases',
      `phase-${phaseNumber}-${phaseNames[phaseNumber]}`,
      `week-${weekNumber}-${weekSlug}`
    );

    const progress = await db.get(
      `SELECT * FROM learning_progress WHERE phase_number = ? AND week_number = ?`,
      [phaseNumber, weekNumber]
    );

    const weekContent = await readWeekContent(contentDir, phaseNumber, weekNumber, progress);

    // Extract specific session data
    const sessionData = weekContent.sessions[sessionType] || null;

    if (!sessionData) {
      return res.status(404).json({
        success: false,
        error: `Session type '${sessionType}' not available for this week`,
      });
    }

    res.json({
      success: true,
      data: {
        session_type: sessionType,
        week_info: {
          phase: phaseNumber,
          week: weekNumber,
          title: weekContent.title,
          description: weekContent.description,
        },
        session_data: sessionData,
        availability: weekContent.session_availability[sessionType] || false,
      },
    });
  } catch (error) {
    console.error('Error fetching session content:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch session content',
      details: error.message,
    });
  }
});

// ==========================================
// PROGRESS TRACKING ROUTES
// ==========================================

// Update session progress
app.post('/api/progress/session', async (req, res) => {
  try {
    const {
      phase,
      week,
      session_type,
      duration_minutes,
      focus_score,
      completion_percentage,
      notes,
    } = req.body;

    // Validate required fields
    if (!phase || !week || !session_type || duration_minutes === undefined) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: phase, week, session_type, duration_minutes',
      });
    }

    // Record daily session
    const today = new Date().toISOString().split('T')[0];
    await db.run(
      `INSERT INTO daily_sessions 
             (session_date, session_type, duration_minutes, focus_score, phase, week, notes, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`,
      [today, session_type, duration_minutes, focus_score || 8, phase, week, notes || null]
    );

    // Update learning progress
    const currentProgress = await db.get(
      `SELECT * FROM learning_progress WHERE phase_number = ? AND week_number = ?`,
      [phase, week]
    );

    const newProgress = Math.min(
      100,
      (currentProgress?.progress_percentage || 0) + (completion_percentage || 25)
    );
    const newStatus =
      newProgress >= 100 ? 'completed' : newProgress > 0 ? 'in_progress' : 'not_started';

    await db.run(
      `INSERT OR REPLACE INTO learning_progress 
             (phase_number, week_number, status, progress_percentage, last_accessed, updated_at)
             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [phase, week, newStatus, newProgress]
    );

    res.json({
      success: true,
      message: 'Session progress recorded successfully',
      data: {
        new_progress: newProgress,
        new_status: newStatus,
      },
    });
  } catch (error) {
    console.error('Error recording session progress:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to record session progress',
      details: error.message,
    });
  }
});

// Get user progress summary
app.get('/api/progress/summary', async (req, res) => {
  try {
    // Get overall progress
    const progressSummary = await db.all(`
            SELECT 
                phase_number,
                week_number,
                status,
                progress_percentage,
                last_accessed
            FROM learning_progress
            ORDER BY phase_number, week_number
        `);

    // Get recent sessions
    const recentSessions = await db.all(`
            SELECT 
                session_date,
                session_type,
                duration_minutes,
                focus_score,
                phase,
                week
            FROM daily_sessions
            ORDER BY created_at DESC
            LIMIT 10
        `);

    // Calculate statistics
    const totalWeeks = progressSummary.length;
    const completedWeeks = progressSummary.filter(p => p.status === 'completed').length;
    const totalProgress = progressSummary.reduce((sum, p) => sum + (p.progress_percentage || 0), 0);
    const averageProgress = totalWeeks > 0 ? totalProgress / totalWeeks : 0;

    res.json({
      success: true,
      data: {
        overall_stats: {
          total_weeks: totalWeeks,
          completed_weeks: completedWeeks,
          average_progress: averageProgress,
          completion_rate: totalWeeks > 0 ? (completedWeeks / totalWeeks) * 100 : 0,
        },
        progress_by_week: progressSummary,
        recent_sessions: recentSessions,
      },
    });
  } catch (error) {
    console.error('Error fetching progress summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch progress summary',
      details: error.message,
    });
  }
});

// ==========================================
// VAULT ROUTES
// ==========================================

// Get vault items (placeholder for now)
app.get('/api/vault', async (req, res) => {
  try {
    // For now, return empty vault - this would be enhanced later
    res.json({
      success: true,
      data: {
        total_items: 0,
        unlocked_items: 0,
        categories: {
          secret_archives: [],
          controversy_files: [],
          beautiful_mind: [],
        },
      },
    });
  } catch (error) {
    console.error('Error fetching vault:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch vault items',
      details: error.message,
    });
  }
});

// ==========================================
// ERROR HANDLING
// ==========================================

// Global error handler
app.use((err, req, res, next) => {
  console.error(chalk.red('❌ Server Error:'), err);

  const errorMessage = isDevelopment ? err.message : 'Internal server error';
  const errorStack = isDevelopment ? err.stack : undefined;

  res.status(err.status || 500).json({
    success: false,
    error: errorMessage,
    stack: errorStack,
    timestamp: new Date().toISOString(),
    path: req.path,
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Not Found',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: new Date().toISOString(),
  });
});

// ==========================================
// SERVER STARTUP
// ==========================================

async function startServer () {
  try {
    console.log(chalk.blue('🔗 Connecting to database...'));
    await db.connect();

    console.log(chalk.blue('🔨 Initializing database schema...'));
    await db.initializeSchema();

    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(chalk.green('\n🚀 Neural Learning Web Backend Server Started!'));
      console.log(chalk.white('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
      console.log(chalk.cyan(`   🌐 Server: http://localhost:${PORT}`));
      console.log(chalk.cyan(`   🔗 Health: http://localhost:${PORT}/health`));
      console.log(chalk.cyan(`   📊 API: http://localhost:${PORT}/api`));
      console.log(chalk.cyan(`   📁 Content: http://localhost:${PORT}/api/content/phases`));
      console.log(chalk.white('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
      console.log(chalk.gray(`   Environment: ${NODE_ENV}`));
      console.log(chalk.gray(`   Database: SQLite`));
      console.log(chalk.gray('   Content: Reading from /content/phases/ directory'));
      console.log(chalk.gray('   Sessions: math, coding, visual_projects, real_applications'));
      console.log(chalk.gray('   Files: content.json, lesson.md, project.md, visualization.py'));
      console.log(chalk.gray('   CORS: Enabled for frontend development\n'));
    });

    // Graceful shutdown
    process.on('SIGTERM', () => gracefulShutdown(server));
    process.on('SIGINT', () => gracefulShutdown(server));

    return server;
  } catch (error) {
    console.error(chalk.red('❌ Failed to start server:'), error);
    process.exit(1);
  }
}

async function gracefulShutdown (server) {
  console.log(chalk.yellow('\n🛑 Graceful shutdown initiated...'));

  server.close(async () => {
    console.log(chalk.blue('📴 HTTP server closed'));

    try {
      await db.close();
      console.log(chalk.blue('🗄️  Database connection closed'));
    } catch (error) {
      console.error(chalk.red('❌ Error closing database:'), error);
    }

    console.log(chalk.green('✅ Graceful shutdown complete'));
    process.exit(0);
  });

  // Force close after 10 seconds
  setTimeout(() => {
    console.log(chalk.red('⚠️  Forced shutdown after timeout'));
    process.exit(1);
  }, 10000);
}

// Start the server
if (require.main === module) {
  startServer();
}

module.exports = { app, startServer };
