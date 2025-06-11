/**
 * Neural Odyssey Database Configuration
 * 
 * SQLite database connection and helper functions for the Neural Odyssey
 * personal learning companion. Provides a centralized database interface
 * for all backend operations.
 * 
 * Features:
 * - Connection management with error handling
 * - Helper functions for common operations
 * - Query utilities with parameter binding
 * - Transaction support
 * - Connection pooling for concurrent requests
 * 
 * Author: Neural Explorer
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');
const chalk = require('chalk');

// Database configuration
const DB_PATH = path.join(__dirname, '../../data/user-progress.sqlite');
const NODE_ENV = process.env.NODE_ENV || 'development';

// Enable verbose mode in development for better debugging
const Database = NODE_ENV === 'development' ? sqlite3.verbose().Database : sqlite3.Database;

class DatabaseManager {
    constructor() {
        this.db = null;
        this.isConnected = false;
        this.connectionRetries = 0;
        this.maxRetries = 3;
        
        // Initialize connection
        this.connect();
    }

    connect() {
        try {
            // Check if database file exists
            if (!fs.existsSync(DB_PATH)) {
                console.error(chalk.red('❌ Database file not found at:'), DB_PATH);
                console.log(chalk.yellow('💡 Run "npm run init-db" to create the database'));
                throw new Error('Database file not found. Please run database initialization first.');
            }

            // Create database connection
            this.db = new Database(DB_PATH, sqlite3.OPEN_READWRITE, (err) => {
                if (err) {
                    console.error(chalk.red('❌ Failed to connect to database:'), err.message);
                    this.handleConnectionError(err);
                    return;
                }
                
                this.isConnected = true;
                this.connectionRetries = 0;
                
                if (NODE_ENV === 'development') {
                    console.log(chalk.green('✅ Connected to SQLite database'));
                    console.log(chalk.blue('📍 Database path:'), chalk.white(DB_PATH));
                }
                
                // Configure database settings
                this.configurePragmas();
            });

            // Handle database errors
            this.db.on('error', (err) => {
                console.error(chalk.red('Database error:'), err);
                this.isConnected = false;
            });

        } catch (error) {
            console.error(chalk.red('Database initialization failed:'), error.message);
            throw error;
        }
    }

    configurePragmas() {
        // Enable foreign key constraints
        this.db.run("PRAGMA foreign_keys = ON");
        
        // Set journal mode to WAL for better concurrency
        this.db.run("PRAGMA journal_mode = WAL");
        
        // Set synchronous mode for better performance in development
        if (NODE_ENV === 'development') {
            this.db.run("PRAGMA synchronous = NORMAL");
        } else {
            this.db.run("PRAGMA synchronous = FULL");
        }
        
        // Enable automatic index creation
        this.db.run("PRAGMA automatic_index = ON");
        
        // Set cache size (negative value means KB)
        this.db.run("PRAGMA cache_size = -64000"); // 64MB cache
        
        if (NODE_ENV === 'development') {
            console.log(chalk.blue('🔧 Database pragmas configured'));
        }
    }

    handleConnectionError(error) {
        this.isConnected = false;
        
        if (this.connectionRetries < this.maxRetries) {
            this.connectionRetries++;
            console.log(chalk.yellow(`⏳ Retrying database connection (${this.connectionRetries}/${this.maxRetries})...`));
            
            setTimeout(() => {
                this.connect();
            }, 2000 * this.connectionRetries); // Exponential backoff
        } else {
            console.error(chalk.red('❌ Max connection retries exceeded'));
            throw new Error('Unable to establish database connection');
        }
    }

    // Promisify database operations for easier async/await usage
    async query(sql, params = []) {
        return new Promise((resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('Database not connected'));
                return;
            }

            this.db.all(sql, params, (err, rows) => {
                if (err) {
                    console.error(chalk.red('Query error:'), err.message);
                    console.error(chalk.red('SQL:'), sql);
                    console.error(chalk.red('Params:'), params);
                    reject(err);
                    return;
                }
                resolve(rows);
            });
        });
    }

    async get(sql, params = []) {
        return new Promise((resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('Database not connected'));
                return;
            }

            this.db.get(sql, params, (err, row) => {
                if (err) {
                    console.error(chalk.red('Get query error:'), err.message);
                    console.error(chalk.red('SQL:'), sql);
                    console.error(chalk.red('Params:'), params);
                    reject(err);
                    return;
                }
                resolve(row);
            });
        });
    }

    async run(sql, params = []) {
        return new Promise((resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('Database not connected'));
                return;
            }

            this.db.run(sql, params, function(err) {
                if (err) {
                    console.error(chalk.red('Run query error:'), err.message);
                    console.error(chalk.red('SQL:'), sql);
                    console.error(chalk.red('Params:'), params);
                    reject(err);
                    return;
                }
                resolve({
                    lastID: this.lastID,
                    changes: this.changes
                });
            });
        });
    }

    async transaction(operations) {
        return new Promise(async (resolve, reject) => {
            if (!this.isConnected) {
                reject(new Error('Database not connected'));
                return;
            }

            try {
                await this.run('BEGIN TRANSACTION');
                
                const results = [];
                for (const operation of operations) {
                    const result = await this.run(operation.sql, operation.params);
                    results.push(result);
                }
                
                await this.run('COMMIT');
                resolve(results);
                
            } catch (error) {
                await this.run('ROLLBACK');
                console.error(chalk.red('Transaction failed, rolled back:'), error.message);
                reject(error);
            }
        });
    }

    // Helper method for batch inserts
    async batchInsert(tableName, records) {
        if (!records || records.length === 0) {
            return { inserted: 0 };
        }

        const keys = Object.keys(records[0]);
        const placeholders = keys.map(() => '?').join(', ');
        const sql = `INSERT INTO ${tableName} (${keys.join(', ')}) VALUES (${placeholders})`;

        const operations = records.map(record => ({
            sql,
            params: keys.map(key => record[key])
        }));

        const results = await this.transaction(operations);
        return { inserted: results.length };
    }

    // Helper method for upsert operations (insert or update)
    async upsert(tableName, data, conflictColumns) {
        const keys = Object.keys(data);
        const values = Object.values(data);
        const placeholders = keys.map(() => '?').join(', ');
        
        const updateClauses = keys
            .filter(key => !conflictColumns.includes(key))
            .map(key => `${key} = excluded.${key}`)
            .join(', ');

        const sql = `
            INSERT INTO ${tableName} (${keys.join(', ')}) 
            VALUES (${placeholders})
            ON CONFLICT(${conflictColumns.join(', ')}) 
            DO UPDATE SET ${updateClauses}
        `;

        return await this.run(sql, values);
    }

    // Health check method
    async healthCheck() {
        try {
            const result = await this.get('SELECT 1 as health');
            return {
                connected: this.isConnected,
                responding: !!result,
                path: DB_PATH,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            return {
                connected: false,
                responding: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    // Get database statistics
    async getStats() {
        try {
            const tableStats = await this.query(`
                SELECT name, 
                       (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as exists
                FROM sqlite_master m 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            `);

            const dbSize = fs.statSync(DB_PATH).size;
            
            return {
                tables: tableStats,
                databaseSize: `${(dbSize / 1024 / 1024).toFixed(2)} MB`,
                filePath: DB_PATH,
                lastModified: fs.statSync(DB_PATH).mtime
            };
        } catch (error) {
            console.error('Failed to get database stats:', error);
            return null;
        }
    }

    // Close database connection
    close() {
        return new Promise((resolve) => {
            if (this.db) {
                this.db.close((err) => {
                    if (err) {
                        console.error(chalk.red('Error closing database:'), err);
                    } else if (NODE_ENV === 'development') {
                        console.log(chalk.green('✅ Database connection closed'));
                    }
                    this.isConnected = false;
                    resolve();
                });
            } else {
                resolve();
            }
        });
    }
}

// Create singleton instance
const dbManager = new DatabaseManager();

// Export both the manager instance and the raw database connection
// This allows for both high-level operations and direct database access when needed
module.exports = {
    // High-level database manager
    manager: dbManager,
    
    // Direct database access (for compatibility with existing code)
    ...dbManager.db,
    
    // Convenience methods
    query: dbManager.query.bind(dbManager),
    get: dbManager.get.bind(dbManager),
    run: dbManager.run.bind(dbManager),
    transaction: dbManager.transaction.bind(dbManager),
    batchInsert: dbManager.batchInsert.bind(dbManager),
    upsert: dbManager.upsert.bind(dbManager),
    healthCheck: dbManager.healthCheck.bind(dbManager),
    getStats: dbManager.getStats.bind(dbManager),
    close: dbManager.close.bind(dbManager),
    
    // Raw database instance for direct access
    database: dbManager.db,
    
    // Connection status
    get isConnected() {
        return dbManager.isConnected;
    },
    
    // Database path
    get path() {
        return DB_PATH;
    }
};

// Handle process termination
process.on('exit', () => {
    if (dbManager.isConnected) {
        dbManager.close();
    }
});

process.on('SIGINT', () => {
    if (dbManager.isConnected) {
        dbManager.close().then(() => {
            process.exit(0);
        });
    } else {
        process.exit(0);
    }
});

process.on('SIGTERM', () => {
    if (dbManager.isConnected) {
        dbManager.close().then(() => {
            process.exit(0);
        });
    } else {
        process.exit(0);
    }
});