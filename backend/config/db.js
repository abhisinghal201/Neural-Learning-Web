/**
 * Neural Odyssey Database Configuration
 *
 * SQLite database connection and management for the Neural Learning Web platform.
 * Provides async wrappers, transaction support, backup functionality, and health monitoring.
 *
 * Features:
 * - Promise-based SQLite operations
 * - Automatic database initialization
 * - Transaction support with rollback
 * - Database backup and restore
 * - Health monitoring and statistics
 * - Connection pooling for single-user setup
 * - Schema validation and migration support
 *
 * Author: Neural Explorer
 */

const sqlite3 = require('sqlite3').verbose()
const fs = require('fs')
const path = require('path')
const { promisify } = require('util')

// Database configuration
const DB_DIR = path.join(__dirname, '../../data')
const DB_PATH = path.join(DB_DIR, 'user-progress.sqlite')
const SCHEMA_PATH = path.join(__dirname, '../../data/schema.sql')
const BACKUP_DIR = path.join(__dirname, '../../backups')

// Ensure directories exist
;[DB_DIR, BACKUP_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
    console.log(`📁 Created directory: ${dir}`)
  }
})

class DatabaseManager {
  constructor () {
    this.db = null
    this.isConnected = false
    this.path = DB_PATH
    this.connectionPromise = null
  }

  /**
   * Initialize database connection
   * @returns {Promise<sqlite3.Database>}
   */
  async connect () {
    if (this.connectionPromise) {
      return this.connectionPromise
    }

    this.connectionPromise = new Promise((resolve, reject) => {
      // SQLite configuration for optimal performance
      const db = new sqlite3.Database(
        DB_PATH,
        sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE,
        err => {
          if (err) {
            console.error('❌ Failed to connect to database:', err.message)
            reject(err)
            return
          }

          console.log('✅ Connected to SQLite database')
          this.db = db
          this.isConnected = true

          // Configure SQLite for better performance
          this.db.serialize(() => {
            this.db.run('PRAGMA journal_mode = WAL')
            this.db.run('PRAGMA synchronous = NORMAL')
            this.db.run('PRAGMA cache_size = -64000') // 64MB cache
            this.db.run('PRAGMA temp_store = memory')
            this.db.run('PRAGMA mmap_size = 268435456') // 256MB mmap
            this.db.run('PRAGMA foreign_keys = ON')
          })

          resolve(db)
        }
      )

      // Handle database errors
      db.on('error', err => {
        console.error('❌ Database error:', err)
        this.isConnected = false
      })

      db.on('close', () => {
        console.log('📴 Database connection closed')
        this.isConnected = false
      })
    })

    return this.connectionPromise
  }

  /**
   * Ensure database is connected
   * @returns {Promise<sqlite3.Database>}
   */
  async ensureConnection () {
    if (!this.isConnected || !this.db) {
      await this.connect()
    }
    return this.db
  }

  /**
   * Execute a SQL query that returns multiple rows
   * @param {string} sql - SQL query
   * @param {Array} params - Query parameters
   * @returns {Promise<Array>} - Query results
   */
  async all (sql, params = []) {
    await this.ensureConnection()
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) {
          console.error('❌ Database query error:', err.message)
          reject(err)
        } else {
          resolve(rows || [])
        }
      })
    })
  }

  /**
   * Execute a SQL query that returns a single row
   * @param {string} sql - SQL query
   * @param {Array} params - Query parameters
   * @returns {Promise<Object|null>} - Query result
   */
  async get (sql, params = []) {
    await this.ensureConnection()
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) {
          console.error('❌ Database query error:', err.message)
          reject(err)
        } else {
          resolve(row || null)
        }
      })
    })
  }

  /**
   * Execute a SQL query that modifies data (INSERT, UPDATE, DELETE)
   * @param {string} sql - SQL query
   * @param {Array} params - Query parameters
   * @returns {Promise<Object>} - Result with lastID, changes, etc.
   */
  async run (sql, params = []) {
    await this.ensureConnection()
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function (err) {
        if (err) {
          console.error('❌ Database run error:', err.message)
          reject(err)
        } else {
          resolve({
            lastID: this.lastID,
            changes: this.changes,
            sql: sql
          })
        }
      })
    })
  }

  /**
   * Execute multiple queries in a transaction
   * @param {Function} callback - Function that receives the database instance
   * @returns {Promise} - Transaction result
   */
  async transaction (callback) {
    await this.ensureConnection()

    return new Promise((resolve, reject) => {
      this.db.serialize(() => {
        this.db.run('BEGIN TRANSACTION', err => {
          if (err) {
            reject(err)
            return
          }

          try {
            const result = callback(this)

            if (result instanceof Promise) {
              result
                .then(data => {
                  this.db.run('COMMIT', commitErr => {
                    if (commitErr) {
                      reject(commitErr)
                    } else {
                      resolve(data)
                    }
                  })
                })
                .catch(error => {
                  this.db.run('ROLLBACK', () => {
                    reject(error)
                  })
                })
            } else {
              this.db.run('COMMIT', commitErr => {
                if (commitErr) {
                  reject(commitErr)
                } else {
                  resolve(result)
                }
              })
            }
          } catch (error) {
            this.db.run('ROLLBACK', () => {
              reject(error)
            })
          }
        })
      })
    })
  }

  /**
   * Execute a batch of queries with transaction support
   * @param {Array} queries - Array of {sql, params} objects
   * @returns {Promise<Array>} - Array of results
   */
  async batch (queries) {
    return this.transaction(async db => {
      const results = []
      for (const query of queries) {
        const result = await db.run(query.sql, query.params || [])
        results.push(result)
      }
      return results
    })
  }

  /**
   * Initialize database schema
   * @returns {Promise<boolean>} - Success status
   */
  async initializeSchema () {
    try {
      console.log('🔨 Initializing database schema...')

      // Check if schema file exists
      if (!fs.existsSync(SCHEMA_PATH)) {
        console.warn('⚠️ Schema file not found, creating basic tables...')
        await this.createBasicSchema()
        return true
      }

      // Read and execute schema file
      const schema = fs.readFileSync(SCHEMA_PATH, 'utf8')
      const statements = schema
        .split(';')
        .map(stmt => stmt.trim())
        .filter(stmt => stmt && !stmt.startsWith('--'))

      for (const statement of statements) {
        if (statement) {
          await this.run(statement)
        }
      }

      console.log('✅ Database schema initialized successfully')
      return true
    } catch (error) {
      console.error('❌ Failed to initialize schema:', error)
      throw error
    }
  }

  /**
   * Create basic schema if schema.sql is missing
   * @returns {Promise<void>}
   */
  async createBasicSchema () {
    const basicTables = [
      `CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL DEFAULT 'Neural Explorer',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                timezone TEXT DEFAULT 'UTC',
                preferred_session_length INTEGER DEFAULT 25,
                daily_goal_minutes INTEGER DEFAULT 60,
                current_phase INTEGER DEFAULT 1,
                current_week INTEGER DEFAULT 1,
                total_study_minutes INTEGER DEFAULT 0,
                current_streak_days INTEGER DEFAULT 0,
                longest_streak_days INTEGER DEFAULT 0,
                last_activity_date DATE,
                theme_preference TEXT DEFAULT 'dark'
            )`,

      `CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase INTEGER NOT NULL,
                week INTEGER NOT NULL,
                lesson_id TEXT NOT NULL,
                lesson_title TEXT NOT NULL,
                status TEXT CHECK(status IN ('not_started', 'in_progress', 'completed', 'mastered')) DEFAULT 'not_started',
                completion_percentage INTEGER DEFAULT 0,
                time_spent_minutes INTEGER DEFAULT 0,
                completed_at DATETIME,
                mastery_score REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phase, week, lesson_id)
            )`,

      `CREATE TABLE IF NOT EXISTS quest_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quest_id TEXT NOT NULL,
                quest_title TEXT NOT NULL,
                quest_type TEXT NOT NULL,
                phase INTEGER NOT NULL,
                week INTEGER NOT NULL,
                status TEXT CHECK(status IN ('attempted', 'completed', 'mastered')) DEFAULT 'attempted',
                time_to_complete_minutes INTEGER,
                user_code TEXT,
                final_solution TEXT,
                completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`,

      `CREATE TABLE IF NOT EXISTS vault_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                item_type TEXT NOT NULL,
                rarity TEXT DEFAULT 'common',
                unlock_conditions TEXT,
                content_preview TEXT,
                content_full TEXT,
                is_unlocked BOOLEAN DEFAULT 0,
                unlocked_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`,

      `CREATE TABLE IF NOT EXISTS daily_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date DATE NOT NULL,
                session_type TEXT CHECK(session_type IN ('math', 'coding', 'visual_projects', 'real_applications')),
                duration_minutes INTEGER DEFAULT 0,
                focus_score INTEGER CHECK(focus_score BETWEEN 1 AND 10),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`
    ]

    for (const sql of basicTables) {
      await this.run(sql)
    }

    // Insert default user profile
    await this.run(`
            INSERT OR IGNORE INTO user_profile (id, username) 
            VALUES (1, 'Neural Explorer')
        `)

    console.log('✅ Basic schema created successfully')
  }

  /**
   * Get database statistics and health information
   * @returns {Promise<Object>} - Database statistics
   */
  async getStats () {
    try {
      await this.ensureConnection()

      const tables = await this.all(`
                SELECT name, sql FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            `)

      const stats = {
        isConnected: this.isConnected,
        tables: tables.length,
        tableNames: tables.map(t => t.name),
        dbPath: DB_PATH
      }

      // Add file system stats if database file exists
      if (fs.existsSync(DB_PATH)) {
        const dbStats = fs.statSync(DB_PATH)
        stats.dbSize = dbStats.size
        stats.dbSizeMB = (dbStats.size / 1024 / 1024).toFixed(2)
        stats.lastModified = dbStats.mtime
      }

      // Get row counts for each table
      for (const table of tables) {
        try {
          const count = await this.get(
            `SELECT COUNT(*) as count FROM ${table.name}`
          )
          stats[`${table.name}_count`] = count ? count.count : 0
        } catch (err) {
          stats[`${table.name}_count`] = 'error'
        }
      }

      return stats
    } catch (error) {
      console.error('❌ Error getting database stats:', error)
      throw error
    }
  }

  /**
   * Backup database to a timestamped file
   * @param {string} backupDir - Directory to store backup
   * @returns {Promise<string>} - Path to backup file
   */
  async backup (backupDir = BACKUP_DIR) {
    try {
      // Ensure backup directory exists
      if (!fs.existsSync(backupDir)) {
        fs.mkdirSync(backupDir, { recursive: true })
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const backupPath = path.join(
        backupDir,
        `neural-odyssey-backup-${timestamp}.sqlite`
      )

      // Close current connection temporarily for clean backup
      if (this.db) {
        await new Promise(resolve => {
          this.db.close(() => {
            this.isConnected = false
            resolve()
          })
        })
      }

      // Copy database file
      await fs.promises.copyFile(DB_PATH, backupPath)

      // Reconnect
      await this.connect()

      console.log('💾 Database backup created:', backupPath)
      return backupPath
    } catch (error) {
      console.error('❌ Database backup failed:', error)
      // Try to reconnect even if backup failed
      try {
        await this.connect()
      } catch (reconnectError) {
        console.error(
          '❌ Failed to reconnect after backup failure:',
          reconnectError
        )
      }
      throw error
    }
  }

  /**
   * Validate database schema and repair if needed
   * @returns {Promise<Object>} - Validation results
   */
  async validateSchema () {
    try {
      const results = {
        valid: true,
        errors: [],
        warnings: [],
        repaired: []
      }

      // Check for required tables
      const requiredTables = [
        'user_profile',
        'learning_progress',
        'quest_completions',
        'daily_sessions',
        'vault_items'
      ]

      const existingTables = await this.all(`
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            `)

      const existingTableNames = existingTables.map(t => t.name)

      // Check for missing tables
      for (const table of requiredTables) {
        if (!existingTableNames.includes(table)) {
          results.errors.push(`Missing required table: ${table}`)
          results.valid = false
        }
      }

      // Check for orphaned data
      if (
        existingTableNames.includes('quest_completions') &&
        existingTableNames.includes('learning_progress')
      ) {
        const orphanedQuests = await this.all(`
                    SELECT COUNT(*) as count FROM quest_completions qc
                    LEFT JOIN learning_progress lp ON qc.phase = lp.phase AND qc.week = lp.week
                    WHERE lp.id IS NULL
                `)

        if (orphanedQuests[0] && orphanedQuests[0].count > 0) {
          results.warnings.push(
            `Found ${orphanedQuests[0].count} orphaned quest completions`
          )
        }
      }

      // Check for data integrity
      if (existingTableNames.includes('user_profile')) {
        const profileCount = await this.get(
          'SELECT COUNT(*) as count FROM user_profile'
        )
        if (!profileCount || profileCount.count === 0) {
          results.warnings.push('No user profile found - creating default')
          await this.run(`
                        INSERT INTO user_profile (id, username) 
                        VALUES (1, 'Neural Explorer')
                    `)
          results.repaired.push('Created default user profile')
        }
      }

      return results
    } catch (error) {
      console.error('❌ Schema validation failed:', error)
      return {
        valid: false,
        errors: [error.message],
        warnings: [],
        repaired: []
      }
    }
  }

  /**
   * Clean up old backup files (keep last 10)
   * @returns {Promise<number>} - Number of files cleaned up
   */
  async cleanupBackups () {
    try {
      if (!fs.existsSync(BACKUP_DIR)) {
        return 0
      }

      const files = fs
        .readdirSync(BACKUP_DIR)
        .filter(
          file =>
            file.startsWith('neural-odyssey-backup-') &&
            file.endsWith('.sqlite')
        )
        .map(file => ({
          name: file,
          path: path.join(BACKUP_DIR, file),
          mtime: fs.statSync(path.join(BACKUP_DIR, file)).mtime
        }))
        .sort((a, b) => b.mtime - a.mtime)

      const filesToDelete = files.slice(10) // Keep newest 10

      for (const file of filesToDelete) {
        fs.unlinkSync(file.path)
      }

      if (filesToDelete.length > 0) {
        console.log(`🗑️ Cleaned up ${filesToDelete.length} old backup files`)
      }

      return filesToDelete.length
    } catch (error) {
      console.error('❌ Backup cleanup failed:', error)
      return 0
    }
  }

  /**
   * Close database connection
   * @returns {Promise<void>}
   */
  async close () {
    return new Promise(resolve => {
      if (this.db) {
        this.db.close(err => {
          if (err) {
            console.error('❌ Error closing database:', err)
          } else {
            console.log('✅ Database connection closed')
          }
          this.isConnected = false
          this.db = null
          this.connectionPromise = null
          resolve()
        })
      } else {
        resolve()
      }
    })
  }

  /**
   * Health check for the database
   * @returns {Promise<Object>} - Health status
   */
  async healthCheck () {
    try {
      await this.ensureConnection()

      // Test basic operations
      const testResult = await this.get('SELECT 1 as test')
      const stats = await this.getStats()

      return {
        status: 'healthy',
        connected: this.isConnected,
        testQuery: testResult ? 'passed' : 'failed',
        tables: stats.tables,
        dbSize: stats.dbSizeMB + ' MB',
        path: DB_PATH
      }
    } catch (error) {
      return {
        status: 'unhealthy',
        connected: false,
        error: error.message,
        path: DB_PATH
      }
    }
  }
}

// Create singleton instance
const dbManager = new DatabaseManager()

// Initialize database on module load
;(async () => {
  try {
    await dbManager.connect()
    await dbManager.initializeSchema()

    // Run validation
    const validation = await dbManager.validateSchema()
    if (!validation.valid) {
      console.warn('⚠️ Database schema issues detected:', validation.errors)
    }

    if (validation.repaired.length > 0) {
      console.log('🔧 Repairs made:', validation.repaired)
    }
  } catch (error) {
    console.error('❌ Failed to initialize database:', error)
  }
})()

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down database connection...')
  await dbManager.close()
  process.exit(0)
})

process.on('SIGTERM', async () => {
  console.log('\n🛑 Shutting down database connection...')
  await dbManager.close()
  process.exit(0)
})

// Export the database manager instance
module.exports = dbManager
