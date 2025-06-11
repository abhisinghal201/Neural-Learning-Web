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
const authRoutes = require('./routes/auth');
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
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"],
        },
    },
    crossOriginEmbedderPolicy: false, // Allow embedding for development
}));

// CORS Configuration
const corsOptions = {
    origin: function (origin, callback) {
        // Allow requests with no origin (mobile apps, Postman, etc.)
        if (!origin) return callback(null, true);
        
        // Allow localhost during development
        if (NODE_ENV === 'development') {
            return callback(null, true);
        }
        
        // In production, you might want to restrict origins
        const allowedOrigins = [
            'http://localhost:3000',
            'http://localhost:5173', // Vite default
            'http://127.0.0.1:3000',
            'http://127.0.0.1:5173'
        ];
        
        if (allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

app.use(cors(corsOptions));

// Rate Limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: NODE_ENV === 'development' ? 1000 : 100, // Generous limit for development
    message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: '15 minutes'
    },
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);

// Compression
app.use(compression());

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

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
            total: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)}MB`
        },
        environment: NODE_ENV,
        version: require('./package.json').version
    });
});

// API Information endpoint
app.get('/api', (req, res) => {
    res.json({
        name: 'Neural Odyssey API',
        version: '1.0.0',
        description: 'Personal ML Learning Companion API',
        endpoints: {
            auth: `${API_VERSION}/auth`,
            learning: `${API_VERSION}/learning`,
            vault: `${API_VERSION}/vault`
        },
        documentation: 'https://github.com/your-username/neural-odyssey/docs',
        status: 'operational'
    });
});

// API Routes
app.use(`${API_VERSION}/auth`, authRoutes);
app.use(`${API_VERSION}/learning`, learningRoutes);
app.use(`${API_VERSION}/vault`, vaultRoutes);

// Serve static files for uploaded content (if any)
const uploadsDir = path.join(__dirname, '../uploads');
if (fs.existsSync(uploadsDir)) {
    app.use('/uploads', express.static(uploadsDir));
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
                else resolve(row);
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
                else resolve(rows);
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
                else resolve(row);
            });
        });

        res.json({
            learning: analytics,
            skills: skillPoints,
            vault: vaultStats,
            generatedAt: new Date().toISOString()
        });

    } catch (error) {
        next(error);
    }
});

// Database status endpoint
app.get(`${API_VERSION}/db/status`, (req, res, next) => {
    db.get("SELECT COUNT(*) as lesson_count FROM learning_progress", (err, row) => {
        if (err) {
            return next(err);
        }
        
        res.json({
            connected: true,
            lessons_loaded: row.lesson_count,
            database_file: path.join(__dirname, '../data/user-progress.sqlite'),
            last_backup: 'Not implemented yet' // Will be updated when backup system is ready
        });
    });
});

// 404 handler for API routes
app.use('/api/*', (req, res) => {
    res.status(404).json({
        error: 'API endpoint not found',
        message: `The endpoint ${req.method} ${req.originalUrl} does not exist`,
        availableEndpoints: [
            'GET /api',
            'GET /health',
            `GET ${API_VERSION}/learning/progress`,
            `GET ${API_VERSION}/vault/items`,
            `GET ${API_VERSION}/analytics/summary`
        ]
    });
});

// Global error handling middleware (must be last)
app.use(errorHandler);

// Graceful shutdown handling
process.on('SIGTERM', () => {
    console.log(chalk.yellow('SIGTERM received. Shutting down gracefully...'));
    
    // Close database connection
    db.close((err) => {
        if (err) {
            console.error(chalk.red('Error closing database:'), err);
        } else {
            console.log(chalk.green('Database connection closed.'));
        }
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log(chalk.yellow('\nSIGINT received. Shutting down gracefully...'));
    
    // Close database connection
    db.close((err) => {
        if (err) {
            console.error(chalk.red('Error closing database:'), err);
        } else {
            console.log(chalk.green('Database connection closed.'));
        }
        process.exit(0);
    });
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error(chalk.red('Uncaught Exception:'), err);
    process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
    console.error(chalk.red('Unhandled Promise Rejection at:'), promise, chalk.red('reason:'), err);
    process.exit(1);
});

// Start server
const server = app.listen(PORT, () => {
    console.log('\n' + chalk.cyan('🚀 Neural Odyssey Backend Server Started'));
    console.log(chalk.green('📍 Server running on:'), chalk.white(`http://localhost:${PORT}`));
    console.log(chalk.green('🔗 API Base URL:'), chalk.white(`http://localhost:${PORT}${API_VERSION}`));
    console.log(chalk.green('🏥 Health Check:'), chalk.white(`http://localhost:${PORT}/health`));
    console.log(chalk.green('🌍 Environment:'), chalk.white(NODE_ENV));
    console.log(chalk.green('📊 Analytics:'), chalk.white(`http://localhost:${PORT}${API_VERSION}/analytics/summary`));
    
    if (NODE_ENV === 'development') {
        console.log(chalk.yellow('\n💡 Development Mode:'));
        console.log(chalk.white('   - CORS allows all origins'));
        console.log(chalk.white('   - Generous rate limiting'));
        console.log(chalk.white('   - Detailed error messages'));
        console.log(chalk.white('   - Use nodemon for auto-restart'));
    }
    
    console.log(chalk.cyan('\n🎯 Ready for your Neural Odyssey journey!\n'));
});

// Export for testing
module.exports = { app, server };