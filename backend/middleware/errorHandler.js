/**
 * Neural Odyssey - Global Error Handling Middleware
 *
 * Comprehensive error handling system that provides:
 * - Structured error responses
 * - Request/response logging
 * - Error categorization and severity levels
 * - Development vs production error details
 * - Database error handling
 * - Validation error formatting
 * - Rate limiting error responses
 *
 * Author: Neural Explorer
 * Version: 1.0.0
 */

const fs = require('fs')
const path = require('path')

// Error logging configuration
const LOG_DIR = path.join(__dirname, '../../logs')
const ERROR_LOG_PATH = path.join(LOG_DIR, 'errors.log')

// Ensure logs directory exists
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true })
}

/**
 * Log error to file with timestamp and context
 * @param {Error} error - The error object
 * @param {Object} req - Express request object
 * @param {string} severity - Error severity level
 */
function logError (error, req, severity = 'ERROR') {
  const timestamp = new Date().toISOString()
  const logEntry = {
    timestamp,
    severity,
    message: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    body: req.method !== 'GET' ? req.body : undefined,
    query: Object.keys(req.query).length > 0 ? req.query : undefined
  }

  const logLine = JSON.stringify(logEntry) + '\n'

  try {
    fs.appendFileSync(ERROR_LOG_PATH, logLine)
  } catch (logError) {
    console.error('❌ Failed to write to error log:', logError)
  }

  // Also log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.error(`\n🔥 ${severity} Error:`, error.message)
    console.error('📍 Request:', req.method, req.url)
    if (error.stack) {
      console.error('📋 Stack:', error.stack)
    }
  }
}

/**
 * Determine error type and appropriate response
 * @param {Error} error - The error object
 * @returns {Object} - Error classification and response data
 */
function classifyError (error) {
  // Database errors
  if (error.code === 'SQLITE_ERROR' || error.message.includes('SQLITE')) {
    return {
      type: 'DATABASE_ERROR',
      statusCode: 500,
      userMessage: 'A database error occurred. Please try again.',
      severity: 'ERROR'
    }
  }

  // Validation errors (express-validator)
  if (error.type === 'ValidationError' || error.array) {
    return {
      type: 'VALIDATION_ERROR',
      statusCode: 400,
      userMessage: 'Invalid input data provided.',
      severity: 'WARNING'
    }
  }

  // File system errors
  if (error.code === 'ENOENT') {
    return {
      type: 'FILE_NOT_FOUND',
      statusCode: 404,
      userMessage: 'Requested resource not found.',
      severity: 'WARNING'
    }
  }

  if (error.code === 'EACCES') {
    return {
      type: 'PERMISSION_ERROR',
      statusCode: 403,
      userMessage: 'Access denied to requested resource.',
      severity: 'ERROR'
    }
  }

  // Rate limiting errors
  if (error.type === 'RateLimitError') {
    return {
      type: 'RATE_LIMIT_ERROR',
      statusCode: 429,
      userMessage: 'Too many requests. Please slow down.',
      severity: 'WARNING'
    }
  }

  // Network/timeout errors
  if (error.code === 'ETIMEDOUT' || error.code === 'ECONNRESET') {
    return {
      type: 'NETWORK_ERROR',
      statusCode: 503,
      userMessage: 'Service temporarily unavailable. Please try again.',
      severity: 'WARNING'
    }
  }

  // JSON parsing errors
  if (error instanceof SyntaxError && error.message.includes('JSON')) {
    return {
      type: 'JSON_PARSE_ERROR',
      statusCode: 400,
      userMessage: 'Invalid JSON format in request body.',
      severity: 'WARNING'
    }
  }

  // Custom application errors
  if (error.name === 'QuestNotFoundError') {
    return {
      type: 'QUEST_NOT_FOUND',
      statusCode: 404,
      userMessage: 'Quest not found or not available.',
      severity: 'INFO'
    }
  }

  if (error.name === 'VaultItemLockedError') {
    return {
      type: 'VAULT_ITEM_LOCKED',
      statusCode: 403,
      userMessage: 'Vault item is locked. Complete required quests to unlock.',
      severity: 'INFO'
    }
  }

  if (error.name === 'ProgressNotFoundError') {
    return {
      type: 'PROGRESS_NOT_FOUND',
      statusCode: 404,
      userMessage: 'Learning progress not found for specified lesson.',
      severity: 'INFO'
    }
  }

  // Authentication/authorization errors
  if (error.name === 'UnauthorizedError') {
    return {
      type: 'UNAUTHORIZED',
      statusCode: 401,
      userMessage: 'Authentication required.',
      severity: 'WARNING'
    }
  }

  if (error.name === 'ForbiddenError') {
    return {
      type: 'FORBIDDEN',
      statusCode: 403,
      userMessage: 'Insufficient permissions.',
      severity: 'WARNING'
    }
  }

  // Default to internal server error
  return {
    type: 'INTERNAL_SERVER_ERROR',
    statusCode: 500,
    userMessage: 'An unexpected error occurred. Please try again.',
    severity: 'ERROR'
  }
}

/**
 * Format error response based on environment
 * @param {Error} error - The error object
 * @param {Object} classification - Error classification
 * @param {Object} req - Express request object
 * @returns {Object} - Formatted error response
 */
function formatErrorResponse (error, classification, req) {
  const isDevelopment = process.env.NODE_ENV === 'development'
  const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

  const baseResponse = {
    success: false,
    error: {
      id: errorId,
      type: classification.type,
      message: classification.userMessage,
      timestamp: new Date().toISOString()
    },
    request: {
      method: req.method,
      url: req.url,
      timestamp: new Date().toISOString()
    }
  }

  // Add development-specific details
  if (isDevelopment) {
    baseResponse.error.details = {
      originalMessage: error.message,
      stack: error.stack,
      code: error.code,
      name: error.name
    }

    if (req.body && Object.keys(req.body).length > 0) {
      baseResponse.request.body = req.body
    }

    if (req.query && Object.keys(req.query).length > 0) {
      baseResponse.request.query = req.query
    }
  }

  // Add validation errors if applicable
  if (error.array && typeof error.array === 'function') {
    baseResponse.error.validationErrors = error.array().map(err => ({
      field: err.param,
      message: err.msg,
      value: err.value,
      location: err.location
    }))
  }

  // Add retry information for temporary errors
  if ([503, 429].includes(classification.statusCode)) {
    baseResponse.error.retryAfter =
      classification.statusCode === 429 ? '15 minutes' : '1 minute'
    baseResponse.error.retryable = true
  }

  return baseResponse
}

/**
 * Main error handling middleware
 * @param {Error} err - Error object
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
function errorHandler (err, req, res, next) {
  // Prevent multiple error responses
  if (res.headersSent) {
    return next(err)
  }

  // Classify the error
  const classification = classifyError(err)

  // Log the error
  logError(err, req, classification.severity)

  // Format the response
  const errorResponse = formatErrorResponse(err, classification, req)

  // Set appropriate headers
  res.set({
    'Content-Type': 'application/json',
    'X-Error-Id': errorResponse.error.id,
    'X-Error-Type': classification.type
  })

  // Send error response
  res.status(classification.statusCode).json(errorResponse)
}

/**
 * Handle async errors in route handlers
 * @param {Function} fn - Async route handler function
 * @returns {Function} - Wrapped function with error handling
 */
function asyncErrorHandler (fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next)
  }
}

/**
 * Custom error classes for application-specific errors
 */
class QuestNotFoundError extends Error {
  constructor (questId) {
    super(`Quest not found: ${questId}`)
    this.name = 'QuestNotFoundError'
    this.questId = questId
  }
}

class VaultItemLockedError extends Error {
  constructor (itemId, requirements) {
    super(
      `Vault item ${itemId} is locked. Requirements: ${requirements.join(', ')}`
    )
    this.name = 'VaultItemLockedError'
    this.itemId = itemId
    this.requirements = requirements
  }
}

class ProgressNotFoundError extends Error {
  constructor (lessonId) {
    super(`Progress not found for lesson: ${lessonId}`)
    this.name = 'ProgressNotFoundError'
    this.lessonId = lessonId
  }
}

class UnauthorizedError extends Error {
  constructor (message = 'Authentication required') {
    super(message)
    this.name = 'UnauthorizedError'
  }
}

class ForbiddenError extends Error {
  constructor (message = 'Insufficient permissions') {
    super(message)
    this.name = 'ForbiddenError'
  }
}

/**
 * Get error statistics for monitoring
 * @returns {Promise<Object>} - Error statistics
 */
async function getErrorStats () {
  try {
    if (!fs.existsSync(ERROR_LOG_PATH)) {
      return {
        totalErrors: 0,
        errorsByType: {},
        errorsBySeverity: {},
        recentErrors: []
      }
    }

    const logContent = fs.readFileSync(ERROR_LOG_PATH, 'utf8')
    const logLines = logContent
      .trim()
      .split('\n')
      .filter(line => line)

    const errors = logLines
      .map(line => {
        try {
          return JSON.parse(line)
        } catch {
          return null
        }
      })
      .filter(Boolean)

    const stats = {
      totalErrors: errors.length,
      errorsByType: {},
      errorsBySeverity: {},
      recentErrors: errors.slice(-10).reverse()
    }

    // Count by type and severity
    errors.forEach(error => {
      const type = error.message.split(':')[0] || 'Unknown'
      const severity = error.severity || 'Unknown'

      stats.errorsByType[type] = (stats.errorsByType[type] || 0) + 1
      stats.errorsBySeverity[severity] =
        (stats.errorsBySeverity[severity] || 0) + 1
    })

    return stats
  } catch (error) {
    console.error('❌ Failed to get error stats:', error)
    return {
      totalErrors: 0,
      errorsByType: {},
      errorsBySeverity: {},
      recentErrors: [],
      error: 'Failed to read error log'
    }
  }
}

module.exports = {
  errorHandler,
  asyncErrorHandler,
  QuestNotFoundError,
  VaultItemLockedError,
  ProgressNotFoundError,
  UnauthorizedError,
  ForbiddenError,
  getErrorStats,
  logError
}
