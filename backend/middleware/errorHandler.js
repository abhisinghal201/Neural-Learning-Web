/**
 * Neural Odyssey Error Handler Middleware
 * 
 * Centralized error handling for the Neural Odyssey backend API.
 * Provides consistent error responses, logging, and development debugging.
 * 
 * Author: Neural Explorer
 */

const chalk = require('chalk');

class AppError extends Error {
    constructor(message, statusCode) {
        super(message);
        this.statusCode = statusCode;
        this.status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
        this.isOperational = true;

        Error.captureStackTrace(this, this.constructor);
    }
}

const handleCastErrorDB = (err) => {
    const message = `Invalid ${err.path}: ${err.value}`;
    return new AppError(message, 400);
};

const handleDuplicateFieldsDB = (err) => {
    const value = err.errmsg.match(/(["'])(\\?.)*?\1/)[0];
    const message = `Duplicate field value: ${value}. Please use another value!`;
    return new AppError(message, 400);
};

const handleValidationErrorDB = (err) => {
    const errors = Object.values(err.errors).map(el => el.message);
    const message = `Invalid input data. ${errors.join('. ')}`;
    return new AppError(message, 400);
};

const handleJWTError = () =>
    new AppError('Invalid token. Please log in again!', 401);

const handleJWTExpiredError = () =>
    new AppError('Your token has expired! Please log in again.', 401);

const handleSQLiteConstraintError = (err) => {
    if (err.message.includes('UNIQUE constraint failed')) {
        return new AppError('This record already exists', 409);
    }
    if (err.message.includes('FOREIGN KEY constraint failed')) {
        return new AppError('Invalid reference to related record', 400);
    }
    if (err.message.includes('NOT NULL constraint failed')) {
        const field = err.message.split('.').pop();
        return new AppError(`Required field missing: ${field}`, 400);
    }
    if (err.message.includes('CHECK constraint failed')) {
        return new AppError('Invalid data format or value', 400);
    }
    return new AppError('Database constraint error', 400);
};

const handleSQLiteError = (err) => {
    if (err.code === 'SQLITE_CONSTRAINT') {
        return handleSQLiteConstraintError(err);
    }
    if (err.code === 'SQLITE_BUSY') {
        return new AppError('Database is busy, please try again', 503);
    }
    if (err.code === 'SQLITE_LOCKED') {
        return new AppError('Database is locked, please try again', 503);
    }
    if (err.code === 'SQLITE_CORRUPT') {
        return new AppError('Database corruption detected', 500);
    }
    if (err.code === 'SQLITE_CANTOPEN') {
        return new AppError('Cannot access database file', 500);
    }
    return new AppError('Database error occurred', 500);
};

const sendErrorDev = (err, req, res) => {
    // API error
    if (req.originalUrl.startsWith('/api')) {
        return res.status(err.statusCode).json({
            success: false,
            error: err,
            message: err.message,
            stack: err.stack,
            request: {
                method: req.method,
                url: req.originalUrl,
                headers: req.headers,
                body: req.body,
                params: req.params,
                query: req.query
            }
        });
    }

    // Rendered website error
    console.error('ERROR 💥', err);
    return res.status(err.statusCode).render('error', {
        title: 'Something went wrong!',
        msg: err.message
    });
};

const sendErrorProd = (err, req, res) => {
    // API error
    if (req.originalUrl.startsWith('/api')) {
        // Operational, trusted error: send message to client
        if (err.isOperational) {
            return res.status(err.statusCode).json({
                success: false,
                message: err.message,
                error_code: err.statusCode,
                timestamp: new Date().toISOString()
            });
        }

        // Programming or other unknown error: don't leak error details
        console.error('ERROR 💥', err);
        return res.status(500).json({
            success: false,
            message: 'Something went wrong!',
            error_code: 500,
            timestamp: new Date().toISOString()
        });
    }

    // Rendered website error
    if (err.isOperational) {
        return res.status(err.statusCode).render('error', {
            title: 'Something went wrong!',
            msg: err.message
        });
    }

    // Programming or other unknown error: don't leak error details
    console.error('ERROR 💥', err);
    return res.status(err.statusCode).render('error', {
        title: 'Something went wrong!',
        msg: 'Please try again later.'
    });
};

const logError = (err, req) => {
    const timestamp = new Date().toISOString();
    const method = req.method;
    const url = req.originalUrl;
    const ip = req.ip || req.connection.remoteAddress;
    const userAgent = req.get('User-Agent') || 'Unknown';

    console.log('\n' + chalk.red('🚨 ERROR OCCURRED'));
    console.log(chalk.gray('─'.repeat(60)));
    console.log(chalk.yellow('Time:'), chalk.white(timestamp));
    console.log(chalk.yellow('Request:'), chalk.white(`${method} ${url}`));
    console.log(chalk.yellow('IP:'), chalk.white(ip));
    console.log(chalk.yellow('User-Agent:'), chalk.white(userAgent));
    console.log(chalk.yellow('Error:'), chalk.red(err.message));
    
    if (err.statusCode >= 500) {
        console.log(chalk.yellow('Stack:'), chalk.gray(err.stack));
    }
    
    console.log(chalk.gray('─'.repeat(60)) + '\n');
};

const errorHandler = (err, req, res, next) => {
    err.statusCode = err.statusCode || 500;
    err.status = err.status || 'error';

    // Log error
    logError(err, req);

    // Add request timing if available
    if (req.startTime) {
        const duration = Date.now() - req.startTime;
        console.log(chalk.yellow('Request Duration:'), chalk.white(`${duration}ms`));
    }

    if (process.env.NODE_ENV === 'development') {
        sendErrorDev(err, req, res);
    } else if (process.env.NODE_ENV === 'production') {
        let error = { ...err };
        error.message = err.message;

        // Handle specific database errors
        if (error.name === 'CastError') error = handleCastErrorDB(error);
        if (error.code === 11000) error = handleDuplicateFieldsDB(error);
        if (error.name === 'ValidationError') error = handleValidationErrorDB(error);
        if (error.name === 'JsonWebTokenError') error = handleJWTError();
        if (error.name === 'TokenExpiredError') error = handleJWTExpiredError();
        
        // Handle SQLite specific errors
        if (error.code && error.code.startsWith('SQLITE_')) {
            error = handleSQLiteError(error);
        }

        sendErrorProd(error, req, res);
    }
};

// Async error wrapper
const catchAsync = (fn) => {
    return (req, res, next) => {
        fn(req, res, next).catch(next);
    };
};

// Validation error formatter
const formatValidationErrors = (errors) => {
    return errors.map(error => ({
        field: error.param,
        message: error.msg,
        value: error.value,
        location: error.location
    }));
};

// Rate limit error handler
const handleRateLimitError = (req, res) => {
    const resetTime = new Date(Date.now() + 15 * 60 * 1000); // 15 minutes from now
    
    res.status(429).json({
        success: false,
        message: 'Too many requests from this IP, please try again later.',
        error_code: 429,
        retry_after: '15 minutes',
        reset_time: resetTime.toISOString(),
        timestamp: new Date().toISOString()
    });
};

// 404 handler for API routes
const handle404 = (req, res, next) => {
    const err = new AppError(`Can't find ${req.originalUrl} on this server!`, 404);
    next(err);
};

// Request timeout handler
const handleTimeout = (req, res) => {
    if (!res.headersSent) {
        res.status(408).json({
            success: false,
            message: 'Request timeout',
            error_code: 408,
            timestamp: new Date().toISOString()
        });
    }
};

// Database connection error handler
const handleDatabaseError = (err) => {
    console.error(chalk.red('🔴 Database Error:'), err.message);
    
    if (err.message.includes('ENOENT')) {
        return new AppError('Database file not found. Please run database initialization.', 503);
    }
    
    if (err.message.includes('EACCES')) {
        return new AppError('Database access denied. Check file permissions.', 503);
    }
    
    return new AppError('Database connection failed', 503);
};

// Memory usage monitoring
const monitorMemoryUsage = (req, res, next) => {
    const memoryUsage = process.memoryUsage();
    const memoryThreshold = 1024 * 1024 * 1024; // 1GB
    
    if (memoryUsage.heapUsed > memoryThreshold) {
        console.warn(chalk.yellow('⚠️  High memory usage detected:'), 
                    chalk.white(`${Math.round(memoryUsage.heapUsed / 1024 / 1024)}MB`));
    }
    
    next();
};

// Request size limit handler
const handlePayloadTooLarge = (err, req, res, next) => {
    if (err.type === 'entity.too.large') {
        return res.status(413).json({
            success: false,
            message: 'Request payload too large',
            error_code: 413,
            max_size: '10MB',
            timestamp: new Date().toISOString()
        });
    }
    next(err);
};

// Graceful shutdown error handler
const handleGracefulShutdown = (signal) => {
    console.log(chalk.yellow(`\n📥 Received ${signal}. Starting graceful shutdown...`));
    
    // Close database connections, stop accepting new requests, etc.
    process.exit(0);
};

// Security error handlers
const handleSecurityError = (err, req, res, next) => {
    if (err.message.includes('CORS')) {
        return res.status(403).json({
            success: false,
            message: 'CORS policy violation',
            error_code: 403,
            timestamp: new Date().toISOString()
        });
    }
    
    if (err.message.includes('CSP')) {
        return res.status(403).json({
            success: false,
            message: 'Content Security Policy violation',
            error_code: 403,
            timestamp: new Date().toISOString()
        });
    }
    
    next(err);
};

// Health check for error handler
const healthCheck = () => {
    return {
        error_handler: 'operational',
        node_env: process.env.NODE_ENV || 'development',
        memory_usage: process.memoryUsage(),
        uptime: process.uptime(),
        timestamp: new Date().toISOString()
    };
};

module.exports = {
    AppError,
    errorHandler,
    catchAsync,
    formatValidationErrors,
    handleRateLimitError,
    handle404,
    handleTimeout,
    handleDatabaseError,
    monitorMemoryUsage,
    handlePayloadTooLarge,
    handleGracefulShutdown,
    handleSecurityError,
    healthCheck
};