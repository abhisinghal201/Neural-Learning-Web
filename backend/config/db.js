/**
 * Neural Odyssey - SQLite Database Configuration
 *
 * Provides a robust SQLite database connection with:
 * - Connection pooling and management
 * - Query helpers and prepared statements
 * - Transaction support
 * - Error handling and logging
 * - Database migration utilities
 * - Performance optimization settings
 *
 * Author: Neural Explorer
 * Version: 1.0.0
 */

const sqlite3 = require('sqlite3').verbose()
const path = require('path')
const fs = require('fs')
const { promisify } = require('util')

// Database configuration
const DB_PATH = path.join(__dirname, '../../data/user-progress.sqlite')
const DB_DIR = path.dirname(DB_PATH)

// Ensure data directory exists
if (!fs.existsSync(DB_DIR)) {
  fs.mkdirSync(DB_DIR, { recursive: true })
  console.log('📁 Created data directory:', DB_DIR)
}

// SQLite database connection with optimization settings
const db = new sqlite3.Database(
  DB_PATH,
  sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE,
  err => {
    if (err) {
      console.error('❌ Error opening database:', err.message)
      process.exit(1)
    } else {
      console.log('✅ Connected to SQLite database:', DB_PATH)
    }
  }
)

// Enable foreign keys and set performance optimizations
db.serialize(() => {
  // Enable foreign key constraints
  db.run('PRAGMA foreign_keys = ON')

  // Set journal mode to WAL for better concurrent access
  db.run('PRAGMA journal_mode = WAL')

  // Set synchronous mode for better performance
  db.run('PRAGMA synchronous = NORMAL')

  // Set cache size (negative value means KB)
  db.run('PRAGMA cache_size = -64000') // 64MB cache

  // Set temp store to memory
  db.run('PRAGMA temp_store = MEMORY')

  // Set mmap size for better I/O performance
  db.run('PRAGMA mmap_size = 268435456') // 256MB

  console.log('⚙️ SQLite performance optimizations applied')
})

// Promisify database methods for async/await support
const dbAsync = {
  // Single row queries
  get: promisify(db.get.bind(db)),

  // Multiple row queries
  all: promisify(db.all.bind(db)),

  // Execute queries (INSERT, UPDATE, DELETE)
  run: promisify(db.run.bind(db)),

  // Execute with parameters
  exec: promisify(db.exec.bind(db)),

  // Prepare statements
  prepare: function (sql) {
    const stmt = db.prepare(sql)
    return {
      get: promisify(stmt.get.bind(stmt)),
      all: promisify(stmt.all.bind(stmt)),
      run: promisify(stmt.run.bind(stmt)),
      finalize: promisify(stmt.finalize.bind(stmt)),
      stmt: stmt
    }
  },

  // Close database connection
  close: promisify(db.close.bind(db)),

  // Raw database instance for advanced operations
  raw: db
}

/**
 * Execute a transaction with automatic rollback on error
 * @param {Function} transactionFn - Function containing transaction operations
 * @returns {Promise} - Resolves with transaction result or rejects with error
 */
dbAsync.transaction = async function (transactionFn) {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.run('BEGIN TRANSACTION')

      transactionFn(dbAsync)
        .then(result => {
          db.run('COMMIT', err => {
            if (err) {
              console.error('❌ Transaction commit failed:', err)
              reject(err)
            } else {
              resolve(result)
            }
          })
        })
        .catch(error => {
          db.run('ROLLBACK', rollbackErr => {
            if (rollbackErr) {
              console.error('❌ Transaction rollback failed:', rollbackErr)
            }
            console.error('❌ Transaction failed:', error)
            reject(error)
          })
        })
    })
  })
}

/**
 * Execute multiple queries in a batch with transaction support
 * @param {Array} queries - Array of {sql, params} objects
 * @returns {Promise} - Resolves with array of results
 */
dbAsync.batch = async function (queries) {
  return dbAsync.transaction(async db => {
    const results = []
    for (const query of queries) {
      const result = await db.run(query.sql, query.params || [])
      results.push(result)
    }
    return results
  })
}

/**
 * Get database statistics and health information
 * @returns {Promise<Object>} - Database statistics
 */
dbAsync.getStats = async function () {
  try {
    const tables = await dbAsync.all(`
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        `)

    const stats = {
      tables: tables.length,
      tableNames: tables.map(t => t.name),
      dbSize: fs.statSync(DB_PATH).size,
      dbPath: DB_PATH
    }

    // Get row counts for each table
    for (const table of tables) {
      try {
        const count = await dbAsync.get(
          `SELECT COUNT(*) as count FROM ${table.name}`
        )
        stats[`${table.name}_count`] = count.count
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
dbAsync.backup = async function (
  backupDir = path.join(__dirname, '../../backups')
) {
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

    // Copy database file
    await fs.promises.copyFile(DB_PATH, backupPath)

    console.log('💾 Database backup created:', backupPath)
    return backupPath
  } catch (error) {
    console.error('❌ Database backup failed:', error)
    throw error
  }
}

/**
 * Validate database schema and repair if needed
 * @returns {Promise<Object>} - Validation results
 */
dbAsync.validateSchema = async function () {
  try {
    const results = {
      valid: true,
      errors: [],
      warnings: []
    }

    // Check for required tables
    const requiredTables = [
      'user_profile',
      'learning_progress',
      'quest_completions',
      'daily_sessions',
      'spaced_repetition',
      'knowledge_graph',
      'vault_items'
    ]

    const existingTables = await dbAsync.all(`
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        `)

    const existingTableNames = existingTables.map(t => t.name)

    for (const table of requiredTables) {
      if (!existingTableNames.includes(table)) {
        results.valid = false
        results.errors.push(`Missing required table: ${table}`)
      }
    }

    // Check foreign key integrity
    const pragmaCheck = await dbAsync.get('PRAGMA foreign_key_check')
    if (pragmaCheck) {
      results.warnings.push('Foreign key constraint violations detected')
    }

    // Check database integrity
    const integrityCheck = await dbAsync.get('PRAGMA integrity_check')
    if (integrityCheck.integrity_check !== 'ok') {
      results.valid = false
      results.errors.push(
        `Database integrity check failed: ${integrityCheck.integrity_check}`
      )
    }

    return results
  } catch (error) {
    console.error('❌ Schema validation failed:', error)
    throw error
  }
}

/**
 * Initialize or migrate database schema
 * @param {string} schemaPath - Path to schema SQL file
 * @returns {Promise<void>}
 */
dbAsync.initializeSchema = async function (
  schemaPath = path.join(__dirname, '../../data/schema.sql')
) {
  try {
    if (!fs.existsSync(schemaPath)) {
      throw new Error(`Schema file not found: ${schemaPath}`)
    }

    const schemaSql = fs.readFileSync(schemaPath, 'utf8')
    await dbAsync.exec(schemaSql)

    console.log('✅ Database schema initialized successfully')
  } catch (error) {
    console.error('❌ Schema initialization failed:', error)
    throw error
  }
}

/**
 * Clean up old or unnecessary data
 * @param {Object} options - Cleanup options
 * @returns {Promise<Object>} - Cleanup results
 */
dbAsync.cleanup = async function (options = {}) {
  const { oldSessionDays = 90, oldProgressDays = 365, vacuum = true } = options

  try {
    const results = {
      deletedSessions: 0,
      deletedProgress: 0,
      freedSpace: 0
    }

    // Clean old daily sessions
    const oldSessionDate = new Date()
    oldSessionDate.setDate(oldSessionDate.getDate() - oldSessionDays)

    const deletedSessions = await dbAsync.run(
      `
            DELETE FROM daily_sessions 
            WHERE date < ?
        `,
      [oldSessionDate.toISOString().split('T')[0]]
    )

    results.deletedSessions = deletedSessions.changes

    // Clean old spaced repetition entries
    const oldProgressDate = new Date()
    oldProgressDate.setDate(oldProgressDate.getDate() - oldProgressDays)

    const deletedProgress = await dbAsync.run(
      `
            DELETE FROM spaced_repetition 
            WHERE last_reviewed < ? AND ease_factor < 1.3
        `,
      [oldProgressDate.toISOString()]
    )

    results.deletedProgress = deletedProgress.changes

    // Vacuum database to reclaim space
    if (vacuum) {
      const sizeBefore = fs.statSync(DB_PATH).size
      await dbAsync.exec('VACUUM')
      const sizeAfter = fs.statSync(DB_PATH).size
      results.freedSpace = sizeBefore - sizeAfter
    }

    console.log('🧹 Database cleanup completed:', results)
    return results
  } catch (error) {
    console.error('❌ Database cleanup failed:', error)
    throw error
  }
}

// Handle process termination
process.on('SIGINT', () => {
  console.log('🛑 Closing database connection...')
  db.close(err => {
    if (err) {
      console.error('❌ Error closing database:', err.message)
    } else {
      console.log('✅ Database connection closed')
    }
    process.exit(0)
  })
})

process.on('SIGTERM', () => {
  console.log('🛑 Closing database connection...')
  db.close(err => {
    if (err) {
      console.error('❌ Error closing database:', err.message)
    } else {
      console.log('✅ Database connection closed')
    }
    process.exit(0)
  })
})

module.exports = dbAsync
