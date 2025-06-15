#!/usr/bin/env node

/**
 * Neural Odyssey Database Backup Script
 * 
 * This script creates automated backups of your learning progress database:
 * - Creates timestamped backup files
 * - Compresses backups to save space
 * - Maintains backup rotation (keeps last N backups)
 * - Exports progress reports in multiple formats
 * - Validates backup integrity
 * 
 * Usage:
 *   node scripts/backup-db.js [options]
 * 
 * Options:
 *   --compress           Compress backup with gzip (default: true)
 *   --format=sql|json    Backup format (default: sql)
 *   --keep=N             Keep last N backups (default: 10)
 *   --output=path        Backup directory (default: backups/)
 *   --verify             Verify backup integrity (default: true)
 *   --report             Generate progress report (default: false)
 *   --quiet              Suppress output (default: false)
 * 
 * Backup Schedule Recommendation:
 *   - Daily: After significant learning sessions
 *   - Weekly: Full backup with progress report
 *   - Monthly: Archive backup with compression
 * 
 * Author: Neural Explorer
 */

const fs = require('fs');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const { promisify } = require('util');
const { spawn } = require('child_process');
const zlib = require('zlib');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    compress: true,
    format: 'sql',
    keep: 10,
    output: 'backups',
    verify: true,
    report: false,
    quiet: false
};

// Parse options
args.forEach(arg => {
    if (arg.startsWith('--')) {
        const [key, value] = arg.substring(2).split('=');
        if (value !== undefined) {
            if (key === 'keep') options[key] = parseInt(value);
            else if (key === 'compress' || key === 'verify' || key === 'report' || key === 'quiet') {
                options[key] = value.toLowerCase() === 'true';
            } else {
                options[key] = value;
            }
        } else {
            // Boolean flags
            options[key] = true;
        }
    }
});

// File paths
const DB_PATH = path.join(__dirname, '../data/user-progress.sqlite');
const BACKUP_DIR = path.join(__dirname, '..', options.output);
const TIMESTAMP = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' + 
                  new Date().toISOString().replace(/[:.]/g, '-').split('T')[1].split('.')[0];

class DatabaseBackup {
    constructor() {
        this.dbPath = DB_PATH;
        this.backupDir = BACKUP_DIR;
        this.timestamp = TIMESTAMP;
        this.db = null;
    }

    log(message) {
        if (!options.quiet) {
            console.log(`[${new Date().toISOString()}] ${message}`);
        }
    }

    error(message) {
        console.error(`[${new Date().toISOString()}] ERROR: ${message}`);
    }

    async ensureBackupDirectory() {
        if (!fs.existsSync(this.backupDir)) {
            fs.mkdirSync(this.backupDir, { recursive: true });
            this.log(`✅ Created backup directory: ${this.backupDir}`);
        }
    }

    async connectToDatabase() {
        return new Promise((resolve, reject) => {
            if (!fs.existsSync(this.dbPath)) {
                reject(new Error(`Database not found: ${this.dbPath}`));
                return;
            }

            this.db = new sqlite3.Database(this.dbPath, sqlite3.OPEN_READONLY, (err) => {
                if (err) {
                    reject(new Error(`Failed to connect to database: ${err.message}`));
                    return;
                }
                this.log('✅ Connected to database');
                resolve();
            });
        });
    }

    async closeDatabase() {
        return new Promise((resolve) => {
            if (this.db) {
                this.db.close(() => {
                    this.log('✅ Database connection closed');
                    resolve();
                });
            } else {
                resolve();
            }
        });
    }

    async createSQLBackup() {
        const backupFileName = `neural-odyssey-backup-${this.timestamp}.sql`;
        const backupPath = path.join(this.backupDir, backupFileName);
        
        this.log('📦 Creating SQL backup...');

        return new Promise((resolve, reject) => {
            // SQLite doesn't have a built-in dump command in Node.js
            // We'll create our own SQL dump
            this.generateSQLDump(backupPath)
                .then(() => {
                    this.log(`✅ SQL backup created: ${backupFileName}`);
                    resolve(backupPath);
                })
                .catch(reject);
        });
    }

    async generateSQLDump(outputPath) {
        const writeStream = fs.createWriteStream(outputPath);
        
        // Write header
        writeStream.write(`-- Neural Odyssey Database Backup\n`);
        writeStream.write(`-- Created: ${new Date().toISOString()}\n`);
        writeStream.write(`-- Database: ${this.dbPath}\n\n`);
        writeStream.write(`PRAGMA foreign_keys = OFF;\n\n`);

        // Get all table names
        const tables = await this.getTables();
        
        for (const table of tables) {
            // Get table schema
            const schema = await this.getTableSchema(table);
            writeStream.write(`-- Table: ${table}\n`);
            writeStream.write(`DROP TABLE IF EXISTS ${table};\n`);
            writeStream.write(`${schema};\n\n`);
            
            // Get table data
            const rows = await this.getTableData(table);
            if (rows.length > 0) {
                for (const row of rows) {
                    const columns = Object.keys(row);
                    const values = columns.map(col => {
                        const val = row[col];
                        if (val === null) return 'NULL';
                        if (typeof val === 'string') return `'${val.replace(/'/g, "''")}'`;
                        return val;
                    });
                    
                    writeStream.write(`INSERT INTO ${table} (${columns.join(', ')}) VALUES (${values.join(', ')});\n`);
                }
                writeStream.write('\n');
            }
        }

        writeStream.write(`PRAGMA foreign_keys = ON;\n`);
        writeStream.end();

        return new Promise((resolve, reject) => {
            writeStream.on('finish', resolve);
            writeStream.on('error', reject);
        });
    }

    async getTables() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            `, (err, rows) => {
                if (err) reject(err);
                else resolve(rows.map(row => row.name));
            });
        });
    }

    async getTableSchema(tableName) {
        return new Promise((resolve, reject) => {
            this.db.get(`
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name=?
            `, [tableName], (err, row) => {
                if (err) reject(err);
                else resolve(row ? row.sql : '');
            });
        });
    }

    async getTableData(tableName) {
        return new Promise((resolve, reject) => {
            this.db.all(`SELECT * FROM ${tableName}`, (err, rows) => {
                if (err) reject(err);
                else resolve(rows || []);
            });
        });
    }

    async createJSONBackup() {
        const backupFileName = `neural-odyssey-backup-${this.timestamp}.json`;
        const backupPath = path.join(this.backupDir, backupFileName);
        
        this.log('📦 Creating JSON backup...');

        const backupData = {
            metadata: {
                created: new Date().toISOString(),
                version: '1.0.0',
                source: this.dbPath,
                format: 'json'
            },
            tables: {}
        };

        const tables = await this.getTables();
        
        for (const table of tables) {
            backupData.tables[table] = await this.getTableData(table);
            this.log(`📋 Exported table: ${table} (${backupData.tables[table].length} rows)`);
        }

        fs.writeFileSync(backupPath, JSON.stringify(backupData, null, 2));
        this.log(`✅ JSON backup created: ${backupFileName}`);
        
        return backupPath;
    }

    async createBinaryBackup() {
        const backupFileName = `neural-odyssey-backup-${this.timestamp}.sqlite`;
        const backupPath = path.join(this.backupDir, backupFileName);
        
        this.log('📦 Creating binary backup...');
        
        // Simple file copy for SQLite
        fs.copyFileSync(this.dbPath, backupPath);
        
        this.log(`✅ Binary backup created: ${backupFileName}`);
        return backupPath;
    }

    async compressBackup(filePath) {
        if (!options.compress) return filePath;

        const compressedPath = `${filePath}.gz`;
        this.log('🗜️ Compressing backup...');

        return new Promise((resolve, reject) => {
            const readStream = fs.createReadStream(filePath);
            const writeStream = fs.createWriteStream(compressedPath);
            const gzip = zlib.createGzip();

            readStream.pipe(gzip).pipe(writeStream);

            writeStream.on('finish', () => {
                // Remove uncompressed file
                fs.unlinkSync(filePath);
                
                const originalSize = fs.statSync(filePath).size;
                const compressedSize = fs.statSync(compressedPath).size;
                const ratio = ((1 - compressedSize / originalSize) * 100).toFixed(1);
                
                this.log(`✅ Backup compressed: ${ratio}% reduction`);
                resolve(compressedPath);
            });

            writeStream.on('error', reject);
            readStream.on('error', reject);
        });
    }

    async verifyBackup(backupPath) {
        if (!options.verify) return true;

        this.log('🔍 Verifying backup integrity...');

        try {
            if (backupPath.endsWith('.gz')) {
                // Verify compressed file
                const readStream = fs.createReadStream(backupPath);
                const gunzip = zlib.createGunzip();
                
                return new Promise((resolve, reject) => {
                    readStream.pipe(gunzip);
                    gunzip.on('end', () => {
                        this.log('✅ Backup verification passed');
                        resolve(true);
                    });
                    gunzip.on('error', reject);
                });
            } else if (backupPath.endsWith('.sqlite')) {
                // Verify SQLite file
                return new Promise((resolve, reject) => {
                    const testDb = new sqlite3.Database(backupPath, sqlite3.OPEN_READONLY, (err) => {
                        if (err) {
                            reject(new Error(`Backup verification failed: ${err.message}`));
                            return;
                        }
                        
                        testDb.get('SELECT COUNT(*) as count FROM learning_progress', (err, row) => {
                            testDb.close();
                            if (err) {
                                reject(new Error(`Backup verification failed: ${err.message}`));
                            } else {
                                this.log('✅ Backup verification passed');
                                resolve(true);
                            }
                        });
                    });
                });
            } else {
                // Verify JSON/SQL file exists and is readable
                const content = fs.readFileSync(backupPath, 'utf8');
                if (content.length > 0) {
                    this.log('✅ Backup verification passed');
                    return true;
                } else {
                    throw new Error('Backup file is empty');
                }
            }
        } catch (error) {
            this.error(`Backup verification failed: ${error.message}`);
            return false;
        }
    }

    async generateProgressReport() {
        if (!options.report) return;

        this.log('📊 Generating progress report...');

        const reportData = await this.getProgressData();
        const reportFileName = `progress-report-${this.timestamp}.md`;
        const reportPath = path.join(this.backupDir, reportFileName);

        const report = this.formatProgressReport(reportData);
        fs.writeFileSync(reportPath, report);

        this.log(`✅ Progress report created: ${reportFileName}`);
    }

    async getProgressData() {
        const profile = await new Promise((resolve, reject) => {
            this.db.get('SELECT * FROM user_profile WHERE id = 1', (err, row) => {
                if (err) reject(err);
                else resolve(row || {});
            });
        });

        const progressStats = await new Promise((resolve, reject) => {
            this.db.get(`
                SELECT 
                    COUNT(*) as total_lessons,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_lessons,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                    SUM(time_spent_minutes) as total_study_time,
                    AVG(mastery_score) as avg_mastery_score
                FROM learning_progress
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row || {});
            });
        });

        const skillPoints = await new Promise((resolve, reject) => {
            this.db.all(`
                SELECT category, SUM(points_earned) as total_points
                FROM skill_points
                GROUP BY category
                ORDER BY total_points DESC
            `, (err, rows) => {
                if (err) reject(err);
                else resolve(rows || []);
            });
        });

        const vaultStats = await new Promise((resolve, reject) => {
            this.db.get(`
                SELECT 
                    COUNT(*) as unlocked_items,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as read_items,
                    AVG(user_rating) as avg_rating
                FROM vault_unlocks
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row || {});
            });
        });

        return { profile, progressStats, skillPoints, vaultStats };
    }

    formatProgressReport(data) {
        const { profile, progressStats, skillPoints, vaultStats } = data;
        
        return `# Neural Odyssey Progress Report

**Generated:** ${new Date().toISOString()}
**Neural Explorer:** ${profile.username || 'Unknown'}

## 📈 Learning Progress

- **Current Phase:** ${profile.current_phase || 1}
- **Current Week:** ${profile.current_week || 1}
- **Total Study Time:** ${Math.floor((profile.total_study_minutes || 0) / 60)}h ${(profile.total_study_minutes || 0) % 60}m
- **Current Streak:** ${profile.current_streak_days || 0} days
- **Longest Streak:** ${profile.longest_streak_days || 0} days

## 🎯 Lesson Completion

- **Total Lessons:** ${progressStats.total_lessons || 0}
- **Completed:** ${progressStats.completed_lessons || 0}
- **Mastered:** ${progressStats.mastered_lessons || 0}
- **Completion Rate:** ${progressStats.total_lessons > 0 ? ((progressStats.completed_lessons / progressStats.total_lessons) * 100).toFixed(1) : 0}%
- **Mastery Rate:** ${progressStats.completed_lessons > 0 ? ((progressStats.mastered_lessons / progressStats.completed_lessons) * 100).toFixed(1) : 0}%
- **Average Mastery Score:** ${progressStats.avg_mastery_score ? (progressStats.avg_mastery_score * 100).toFixed(1) : 'N/A'}%

## 🏆 Skill Points

${skillPoints.map(skill => `- **${skill.category.charAt(0).toUpperCase() + skill.category.slice(1)}:** ${skill.total_points} points`).join('\n')}

## 🗝️ Vault Progress

- **Items Unlocked:** ${vaultStats.unlocked_items || 0}
- **Items Read:** ${vaultStats.read_items || 0}
- **Average Rating:** ${vaultStats.avg_rating ? `${vaultStats.avg_rating.toFixed(1)}/5 ⭐` : 'Not rated'}

## 💾 Backup Information

- **Backup Date:** ${new Date().toISOString()}
- **Database Size:** ${this.getFileSize(this.dbPath)}
- **Backup Format:** ${options.format}
- **Compression:** ${options.compress ? 'Enabled' : 'Disabled'}

---

*Keep exploring, Neural Explorer! 🧠✨*
`;
    }

    getFileSize(filePath) {
        const stats = fs.statSync(filePath);
        const bytes = stats.size;
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }

    async cleanupOldBackups() {
        this.log('🧹 Cleaning up old backups...');
        
        const files = fs.readdirSync(this.backupDir)
            .filter(file => file.startsWith('neural-odyssey-backup-'))
            .map(file => ({
                name: file,
                path: path.join(this.backupDir, file),
                stat: fs.statSync(path.join(this.backupDir, file))
            }))
            .sort((a, b) => b.stat.mtime - a.stat.mtime);

        if (files.length > options.keep) {
            const filesToDelete = files.slice(options.keep);
            for (const file of filesToDelete) {
                fs.unlinkSync(file.path);
                this.log(`🗑️ Deleted old backup: ${file.name}`);
            }
        }
    }

    async run() {
        try {
            this.log('🚀 Starting Neural Odyssey database backup...');
            
            await this.ensureBackupDirectory();
            await this.connectToDatabase();
            
            let backupPath;
            
            switch (options.format) {
                case 'json':
                    backupPath = await this.createJSONBackup();
                    break;
                case 'binary':
                    backupPath = await this.createBinaryBackup();
                    break;
                case 'sql':
                default:
                    backupPath = await this.createSQLBackup();
                    break;
            }

            if (options.compress) {
                backupPath = await this.compressBackup(backupPath);
            }

            const isValid = await this.verifyBackup(backupPath);
            if (!isValid) {
                throw new Error('Backup verification failed');
            }

            await this.generateProgressReport();
            await this.cleanupOldBackups();
            await this.closeDatabase();

            const backupSize = this.getFileSize(backupPath);
            this.log(`✅ Backup completed successfully!`);
            this.log(`📁 Backup file: ${path.basename(backupPath)} (${backupSize})`);
            this.log(`📂 Location: ${this.backupDir}`);

        } catch (error) {
            this.error(`Backup failed: ${error.message}`);
            await this.closeDatabase();
            process.exit(1);
        }
    }
}

// Run backup if script is executed directly
if (require.main === module) {
    const backup = new DatabaseBackup();
    backup.run().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = DatabaseBackup;