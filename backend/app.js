/**
 * Neural Odyssey Backend API Server
 * 
 * Main Express.js server for the Neural Odyssey learning platform.
 * Provides RESTful API endpoints for:
 * - Learning progress tracking
 * - Vault item management  
 * - Quest/project completions
 * - User analytics and insights
 * - Spaced repetition system
 * 
 * Single-user configuration - no authentication required
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
const fs = require('fs');
const chalk = require('chalk');

// Load environment variables
require('dotenv').config();

// Import database configuration
const db = require('./config/db');

// Import route handlers
const learningRoutes = require('./routes/learning');
const vaultRoutes = require('./routes/vault');

// Import middleware
const errorHandler = require('./middleware/errorHandler');

// Initialize Express app
const app = express();

// Configuration
const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';
const API_VERSION = '/api/v1';

// Trust proxy for accurate IP addresses (useful for rate limiting)
app.set('trust proxy', 1);

// Security Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            scriptSrc: ["'self'", "'unsafe-eval'", "https://cdnjs.cloudflare.com"], // unsafe-eval for Pyodide
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'", "https://cdnjs.cloudflare.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com"],
            workerSrc: ["'self'", "blob:"], // For Pyodide web workers
        },
    },
    crossOriginEmbedderPolicy: false, // Allow embedding for development
}));

// CORS Configuration for single-user local development
const corsOptions = {
    origin: function (origin, callback) {
        // Allow requests with no origin (mobile apps, Postman, etc.)
        if (!origin) return callback(null, true);
        
        // Allow all localhost during development (single-user environment)
        if (NODE_ENV === 'development') {
            return callback(null, true);
        }
        
        // In production/deployment, allow local origins
        const allowedOrigins = [
            'http://localhost:3000',
            'http://localhost:5173', // Vite default
            'http://127.0.0.1:3000',
            'http://127.0.0.1:5173',
            'http://localhost:4173', // Vite preview
            'http://127.0.0.1:4173'
        ];
        
        if (allowedOrigins.some(allowedOrigin => origin.startsWith(allowedOrigin))) {
            callback(null, true);
        } else {
            callback(null, true); // Allow all for single-user setup
        }
    },
    credentials: false, // No credentials needed for single-user
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'X-Requested-With']
};

app.use(cors(corsOptions));

// Rate Limiting (generous for single-user)
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: NODE_ENV === 'development' ? 10000 : 1000, // Very generous for single user
    message: {
        error: 'Too many requests, please slow down a bit.',
        retryAfter: '15 minutes'
    },
    standardHeaders: true,
    legacyHeaders: false,
    skip: (req) => {
        // Skip rate limiting for health checks and static files
        return req.path === '/health' || req.path.startsWith('/uploads');
    }
});
app.use(limiter);

// Compression
app.use(compression());

// Body parsing middleware
app.use(express.json({ limit: '50mb' })); // Generous for code submissions
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Logging middleware
if (NODE_ENV === 'development') {
    app.use(morgan('dev'));
} else {
    app.use(morgan('combined'));
}

// Request timing middleware
app.use((req, res, next) => {
    req.startTime = Date.now();
    next();
});

// Request ID middleware for debugging
app.use((req, res, next) => {
    req.id = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    res.set('X-Request-ID', req.id);
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    const uptime = process.uptime();
    const memoryUsage = process.memoryUsage();
    
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: `${Math.floor(uptime / 60)}m ${Math.floor(uptime % 60)}s`,
        memory: {
            used: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)}MB`,
            total: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)}MB`,
            rss: `${Math.round(memoryUsage.rss / 1024 / 1024)}MB`
        },
        environment: NODE_ENV,
        version: require('./package.json').version,
        database: {
            connected: db.isConnected || false,
            path: db.path || 'unknown'
        }
    });
});

// API Information endpoint
app.get('/api', (req, res) => {
    res.json({
        name: 'Neural Odyssey API',
        version: '1.0.0',
        description: 'Personal ML Learning Companion API - Single User Edition',
        author: 'Neural Explorer',
        endpoints: {
            learning: `${API_VERSION}/learning`,
            vault: `${API_VERSION}/vault`,
            analytics: `${API_VERSION}/analytics`
        },
        features: [
            'Learning progress tracking',
            'Vault reward system',
            'Quest management',
            'Spaced repetition',
            'Analytics dashboard'
        ],
        documentation: 'https://github.com/abhisinghal21/neural-odyssey/docs',
        status: 'operational'
    });
});

// API Routes (no auth routes needed for single user)
app.use(`${API_VERSION}/learning`, learningRoutes);
app.use(`${API_VERSION}/vault`, vaultRoutes);

// Serve static files for uploaded content (if any)
const uploadsDir = path.join(__dirname, '../uploads');
if (fs.existsSync(uploadsDir)) {
    app.use('/uploads', express.static(uploadsDir, {
        maxAge: '1h', // Cache uploaded files
        etag: false
    }));
}

// Analytics endpoint for learning insights
app.get(`${API_VERSION}/analytics/summary`, async (req, res, next) => {
    try {
        const analytics = await new Promise((resolve, reject) => {
            db.get(`
                SELECT 
                    COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                    AVG(time_spent_minutes) as avg_time_per_lesson,
                    SUM(time_spent_minutes) as total_study_time,
                    COUNT(DISTINCT phase) as active_phases,
                    MAX(updated_at) as last_activity
                FROM learning_progress
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row || {});
            });
        });

        const skillPoints = await new Promise((resolve, reject) => {
            db.all(`
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
            db.get(`
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

        const userProfile = await new Promise((resolve, reject) => {
            db.get(`
                SELECT 
                    current_phase,
                    current_week,
                    current_streak_days,
                    total_study_minutes,
                    neural_explorer_level
                FROM user_profile 
                WHERE id = 1
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row || {});
            });
        });

        res.json({
            success: true,
            data: {
                learning: analytics,
                skills: skillPoints,
                vault: vaultStats,
                profile: userProfile,
                generatedAt: new Date().toISOString(),
                requestId: req.id
            }
        });

    } catch (error) {
        next(error);
    }
});

// Database status endpoint
app.get(`${API_VERSION}/db/status`, async (req, res, next) => {
    try {
        const lessonCount = await new Promise((resolve, reject) => {
            db.get("SELECT COUNT(*) as lesson_count FROM learning_progress", (err, row) => {
                if (err) reject(err);
                else resolve(row?.lesson_count || 0);
            });
        });

        const dbStats = fs.existsSync(db.path) ? fs.statSync(db.path) : null;
        
        res.json({
            success: true,
            data: {
                connected: db.isConnected || false,
                lessons_loaded: lessonCount,
                database_file: db.path,
                database_size: dbStats ? `${(dbStats.size / 1024 / 1024).toFixed(2)} MB` : 'Unknown',
                last_modified: dbStats ? dbStats.mtime : null,
                backup_status: 'Manual', // Will be updated when backup system is implemented
                tables_accessible: true
            }
        });

    } catch (error) {
        next(error);
    }
});

// User profile endpoint (since there's only one user)
app.get(`${API_VERSION}/profile`, async (req, res, next) => {
    try {
        const profile = await new Promise((resolve, reject) => {
            db.get("SELECT * FROM user_profile WHERE id = 1", (err, row) => {
                if (err) reject(err);
                else resolve(row);
            });
        });

        if (!profile) {
            return res.status(404).json({
                success: false,
                message: 'User profile not found. Please run database initialization.'
            });
        }

        res.json({
            success: true,
            data: profile
        });

    } catch (error) {
        next(error);
    }
});

// Update user profile endpoint
app.put(`${API_VERSION}/profile`, async (req, res, next) => {
    try {
        const {
            username,
            timezone,
            preferred_session_length,
            daily_goal_minutes,
            notification_enabled,
            theme_preference,
            animation_enabled,
            sound_enabled
        } = req.body;

        // Build update query dynamically
        const updates = [];
        const values = [];

        if (username !== undefined) {
            updates.push('username = ?');
            values.push(username);
        }
        if (timezone !== undefined) {
            updates.push('timezone = ?');
            values.push(timezone);
        }
        if (preferred_session_length !== undefined) {
            updates.push('preferred_session_length = ?');
            values.push(preferred_session_length);
        }
        if (daily_goal_minutes !== undefined) {
            updates.push('daily_goal_minutes = ?');
            values.push(daily_goal_minutes);
        }
        if (notification_enabled !== undefined) {
            updates.push('notification_enabled = ?');
            values.push(notification_enabled ? 1 : 0);
        }
        if (theme_preference !== undefined) {
            updates.push('theme_preference = ?');
            values.push(theme_preference);
        }
        if (animation_enabled !== undefined) {
            updates.push('animation_enabled = ?');
            values.push(animation_enabled ? 1 : 0);
        }
        if (sound_enabled !== undefined) {
            updates.push('sound_enabled = ?');
            values.push(sound_enabled ? 1 : 0);
        }

        if (updates.length === 0) {
            return res.status(400).json({
                success: false,
                message: 'No valid fields to update'
            });
        }

        updates.push('updated_at = CURRENT_TIMESTAMP');
        values.push(1); // WHERE id = 1

        const sql = `UPDATE user_profile SET ${updates.join(', ')} WHERE id = ?`;
        
        await new Promise((resolve, reject) => {
            db.run(sql, values, function(err) {
                if (err) reject(err);
                else resolve(this);
            });
        });

        // Get updated profile
        const updatedProfile = await new Promise((resolve, reject) => {
            db.get("SELECT * FROM user_profile WHERE id = 1", (err, row) => {
                if (err) reject(err);
                else resolve(row);
            });
        });

        res.json({
            success: true,
            message: 'Profile updated successfully',
            data: updatedProfile
        });

    } catch (error) {
        next(error);
    }
});

// 404 handler for API routes
app.use('/api/*', (req, res) => {
    res.status(404).json({
        success: false,
        error: 'API endpoint not found',
        message: `The endpoint ${req.method} ${req.originalUrl} does not exist`,
        availableEndpoints: [
            'GET /api',
            'GET /health',
            `GET ${API_VERSION}/profile`,
            `PUT ${API_VERSION}/profile`,
            `GET ${API_VERSION}/learning/progress`,
            `GET ${API_VERSION}/vault/items`,
            `GET ${API_VERSION}/analytics/summary`,
            `GET ${API_VERSION}/db/status`
        ],
        requestId: req.id
    });
});

// Global error handling middleware (must be last)
app.use(errorHandler);

// Graceful shutdown handling
const gracefulShutdown = (signal) => {
    console.log(chalk.yellow(`\n${signal} received. Starting graceful shutdown...`));
    
    const shutdownTimeout = setTimeout(() => {
        console.log(chalk.red('Forceful shutdown due to timeout'));
        process.exit(1);
    }, 10000); // 10 second timeout

    // Close database connection
    if (db && db.close) {
        db.close((err) => {
            if (err) {
                console.error(chalk.red('Error closing database:'), err);
            } else {
                console.log(chalk.green('Database connection closed'));
            }
            
            clearTimeout(shutdownTimeout);
            console.log(chalk.green('Graceful shutdown completed'));
            process.exit(0);
        });
    } else {
        clearTimeout(shutdownTimeout);
        console.log(chalk.green('Graceful shutdown completed'));
        process.exit(0);
    }
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error(chalk.red('Uncaught Exception:'), err);
    console.error(chalk.red('Stack:'), err.stack);
    process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
    console.error(chalk.red('Unhandled Promise Rejection at:'), promise);
    console.error(chalk.red('Reason:'), err);
    process.exit(1);
});

// Start server
const server = app.listen(PORT, '127.0.0.1', () => {
    console.log('\n' + chalk.cyan('🚀 Neural Odyssey Backend Server Started'));
    console.log(chalk.green('📍 Server running on:'), chalk.white(`http://127.0.0.1:${PORT}`));
    console.log(chalk.green('🔗 API Base URL:'), chalk.white(`http://127.0.0.1:${PORT}${API_VERSION}`));
    console.log(chalk.green('🏥 Health Check:'), chalk.white(`http://127.0.0.1:${PORT}/health`));
    console.log(chalk.green('🌍 Environment:'), chalk.white(NODE_ENV));
    console.log(chalk.green('📊 Analytics:'), chalk.white(`http://127.0.0.1:${PORT}${API_VERSION}/analytics/summary`));
    console.log(chalk.green('👤 Profile:'), chalk.white(`http://127.0.0.1:${PORT}${API_VERSION}/profile`));
    
    if (NODE_ENV === 'development') {
        console.log(chalk.yellow('\n💡 Development Mode:'));
        console.log(chalk.white('   - CORS allows all localhost origins'));
        console.log(chalk.white('   - Generous rate limiting (10k req/15min)'));
        console.log(chalk.white('   - Detailed error messages'));
        console.log(chalk.white('   - No authentication required'));
        console.log(chalk.white('   - Use nodemon for auto-restart'));
    }
    
    console.log(chalk.cyan('\n🎯 Ready for your Neural Odyssey journey!'));
    console.log(chalk.gray('   Single-user mode: No authentication required'));
    console.log(chalk.gray('   Database: SQLite local storage'));
    console.log(chalk.gray('   Pyodide: Browser-based Python execution\n'));
});

// Handle server errors
server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(chalk.red(`❌ Port ${PORT} is already in use`));
        console.log(chalk.yellow('💡 Try:'));
        console.log(chalk.white(`   - Change PORT in .env file`));
        console.log(chalk.white(`   - Kill process using port: lsof -ti:${PORT} | xargs kill -9`));
        console.log(chalk.white(`   - Use different port: PORT=3002 npm run dev`));
    } else {
        console.error(chalk.red('❌ Server error:'), err);
    }
    process.exit(1);
});

// Export for testing
module.exports = { app, server };