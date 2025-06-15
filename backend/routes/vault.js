/**
 * Neural Odyssey Vault Routes
 *
 * RESTful API endpoints for managing the Neural Vault - the gamification system
 * that rewards learning progress with unlockable content, insights, and treasures.
 *
 * Endpoints:
 * - GET /items - Get all vault items with unlock status
 * - GET /items/:id - Get specific vault item details
 * - POST /unlock/:id - Attempt to unlock a vault item
 * - GET /unlocked - Get unlocked vault items (user's collection)
 * - GET /unlock-conditions/:id - Get unlock requirements for an item
 * - POST /check-unlocks - Check and unlock eligible items based on progress
 * - GET /categories - Get vault items grouped by category
 * - GET /categories/:category - Get vault items in specific category
 * - GET /recent - Get recently unlocked items
 * - GET /favorites - Get user's favorite vault items
 * - GET /statistics - Get vault statistics and progress
 * - POST /items/:id/read - Mark vault item as read
 * - POST /items/:id/rate - Rate a vault item
 * - POST /items/:id/favorite - Toggle favorite status
 * - GET /recommendations - Get personalized vault recommendations
 * - GET /timeline - Get vault unlock timeline
 * - GET /search - Search vault items
 * - GET /analytics - Get detailed vault analytics
 *
 * Author: Neural Explorer
 */

const express = require('express')
const { body, param, query, validationResult } = require('express-validator')
const router = express.Router()
const db = require('../config/db')
const VaultItem = require('../models/VaultItem')
const fs = require('fs')
const path = require('path')
const moment = require('moment')
const {
  asyncErrorHandler,
  VaultItemLockedError
} = require('../middleware/errorHandler')

// Load vault items configuration
const VAULT_ITEMS_PATH = path.join(__dirname, '../../data/vault-items.json')

// Middleware to handle validation errors
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req)
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      error: {
        type: 'VALIDATION_ERROR',
        message: 'Invalid input data provided.',
        details: errors.array()
      }
    })
  }
  next()
}

/**
 * Load vault items from configuration file
 * @returns {Array} - Array of vault item configurations
 */
function loadVaultItemsConfig () {
  try {
    if (fs.existsSync(VAULT_ITEMS_PATH)) {
      const configData = fs.readFileSync(VAULT_ITEMS_PATH, 'utf8')
      return JSON.parse(configData)
    }
    return []
  } catch (error) {
    console.warn('⚠️ Could not load vault items configuration:', error.message)
    return []
  }
}

/**
 * Get user's overall progress for unlock condition checking
 * @returns {Promise<Object>} - User progress data
 */
async function getUserProgress () {
  try {
    const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')
    const lessonProgress = await db.all('SELECT * FROM learning_progress')
    const questProgress = await db.all('SELECT * FROM quest_completions')
    const skillPoints = await db.all('SELECT * FROM skill_points')
    const vaultUnlocks = await db.all('SELECT * FROM vault_unlocks')

    return {
      profile,
      lessons: lessonProgress,
      quests: questProgress,
      skills: skillPoints,
      vault: vaultUnlocks
    }
  } catch (error) {
    console.error('❌ Error getting user progress:', error)
    return null
  }
}

// ==========================================
// MAIN VAULT ENDPOINTS
// ==========================================

// GET /api/v1/vault/items - Get all vault items with unlock status
router.get(
  '/items',
  [
    query('category')
      .optional()
      .isIn(['secret_archives', 'controversy_files', 'beautiful_mind']),
    query('type')
      .optional()
      .isIn(['secret', 'tool', 'artifact', 'story', 'wisdom']),
    query('rarity')
      .optional()
      .isIn(['common', 'uncommon', 'rare', 'epic', 'legendary']),
    query('status').optional().isIn(['locked', 'unlocked', 'read', 'unread']),
    query('difficulty')
      .optional()
      .isIn(['beginner', 'intermediate', 'advanced', 'expert']),
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('offset').optional().isInt({ min: 0 }),
    query('search').optional().isString().isLength({ max: 100 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      category,
      type,
      rarity,
      status,
      difficulty,
      limit = 50,
      offset = 0,
      search
    } = req.query

    // Build criteria object
    const criteria = { limit: parseInt(limit) }

    if (category) criteria.category = category
    if (type) criteria.itemType = type
    if (rarity) criteria.rarity = rarity
    if (difficulty) criteria.difficulty = difficulty

    // Handle status filtering
    if (status === 'unlocked') {
      criteria.isUnlocked = true
    } else if (status === 'locked') {
      criteria.isUnlocked = false
    }

    // Get vault items
    let vaultItems = await VaultItem.findByCriteria(criteria)

    // Apply search filter if provided
    if (search) {
      const searchLower = search.toLowerCase()
      vaultItems = vaultItems.filter(
        item =>
          item.title.toLowerCase().includes(searchLower) ||
          item.description.toLowerCase().includes(searchLower) ||
          item.tags.some(tag => tag.toLowerCase().includes(searchLower))
      )
    }

    // Apply read/unread filtering
    if (status === 'read') {
      vaultItems = vaultItems.filter(item => item.isRead)
    } else if (status === 'unread') {
      vaultItems = vaultItems.filter(item => item.isUnlocked && !item.isRead)
    }

    // Apply offset
    const paginatedItems = vaultItems.slice(
      parseInt(offset),
      parseInt(offset) + parseInt(limit)
    )

    // For locked items, check unlock conditions
    const itemsWithUnlockStatus = []
    for (const item of paginatedItems) {
      if (!item.isUnlocked) {
        const unlockStatus = await item.checkUnlockConditions()
        itemsWithUnlockStatus.push({
          ...item.toJSON(false),
          unlockStatus
        })
      } else {
        itemsWithUnlockStatus.push(item.toJSON(true))
      }
    }

    res.json({
      success: true,
      data: {
        items: itemsWithUnlockStatus,
        pagination: {
          total: vaultItems.length,
          limit: parseInt(limit),
          offset: parseInt(offset),
          hasMore: parseInt(offset) + parseInt(limit) < vaultItems.length
        },
        filters: {
          category,
          type,
          rarity,
          status,
          difficulty,
          search
        }
      }
    })
  })
)

// GET /api/v1/vault/items/:id - Get specific vault item details
router.get(
  '/items/:id',
  [param('id').notEmpty().withMessage('Item ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const vaultItem = await VaultItem.findById(id)

    if (!vaultItem) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    // Get unlock status and related items
    let unlockStatus = null
    if (!vaultItem.isUnlocked) {
      unlockStatus = await vaultItem.checkUnlockConditions()
    }

    const relatedItems = await vaultItem.getRelatedItems()
    const statistics = await vaultItem.getStatistics()

    res.json({
      success: true,
      data: {
        item: vaultItem.toJSON(vaultItem.isUnlocked),
        unlockStatus,
        relatedItems: relatedItems.map(item => item.toJSON(false)),
        statistics
      }
    })
  })
)

// POST /api/v1/vault/unlock/:id - Attempt to unlock a vault item
router.post(
  '/unlock/:id',
  [param('id').notEmpty().withMessage('Item ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const vaultItem = await VaultItem.findById(id)

    if (!vaultItem) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    // Check if already unlocked
    if (vaultItem.isUnlocked) {
      return res.json({
        success: true,
        message: 'Item is already unlocked!',
        data: {
          item: vaultItem.toJSON(true),
          alreadyUnlocked: true
        }
      })
    }

    try {
      // Attempt to unlock
      const unlockResult = await vaultItem.unlock('manual_unlock')

      res.json({
        success: true,
        message: unlockResult.message,
        data: {
          item: vaultItem.toJSON(true),
          bonusPoints: unlockResult.bonusPoints,
          trigger: unlockResult.trigger
        }
      })
    } catch (error) {
      if (error.message.includes('Cannot unlock vault item')) {
        // Get detailed unlock status for error response
        const unlockStatus = await vaultItem.checkUnlockConditions()

        return res.status(403).json({
          success: false,
          error: {
            type: 'VAULT_ITEM_LOCKED',
            message: 'Vault item cannot be unlocked yet.',
            missingRequirements: unlockStatus.missingRequirements,
            progress: unlockStatus.progress
          }
        })
      }
      throw error
    }
  })
)

// GET /api/v1/vault/unlocked - Get unlocked vault items (user's collection)
router.get(
  '/unlocked',
  [
    query('category')
      .optional()
      .isIn(['secret_archives', 'controversy_files', 'beautiful_mind']),
    query('sort').optional().isIn(['recent', 'title', 'rating', 'read_time']),
    query('limit').optional().isInt({ min: 1, max: 100 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { category, sort = 'recent', limit = 50 } = req.query

    const criteria = {
      isUnlocked: true,
      limit: parseInt(limit)
    }

    if (category) {
      criteria.category = category
    }

    // Set order based on sort parameter
    switch (sort) {
      case 'title':
        criteria.orderBy = 'vi.title ASC'
        break
      case 'rating':
        criteria.orderBy = 'vu.user_rating DESC, vi.title ASC'
        break
      case 'read_time':
        criteria.orderBy = 'vi.estimated_read_time ASC'
        break
      default: // recent
        criteria.orderBy = 'vu.unlocked_at DESC'
    }

    const unlockedItems = await VaultItem.findByCriteria(criteria)

    // Group by category for better organization
    const itemsByCategory = {
      secret_archives: [],
      controversy_files: [],
      beautiful_mind: []
    }

    unlockedItems.forEach(item => {
      itemsByCategory[item.category].push(item.toJSON(true))
    })

    // Calculate collection statistics
    const totalUnlocked = unlockedItems.length
    const readItems = unlockedItems.filter(item => item.isRead).length
    const favoriteItems = unlockedItems.filter(item => item.isFavorite).length
    const averageRating = unlockedItems
      .filter(item => item.userRating)
      .reduce((sum, item, _, arr) => sum + item.userRating / arr.length, 0)

    res.json({
      success: true,
      data: {
        items: unlockedItems.map(item => item.toJSON(true)),
        itemsByCategory,
        statistics: {
          totalUnlocked,
          readItems,
          favoriteItems,
          readRate: totalUnlocked > 0 ? readItems / totalUnlocked : 0,
          averageRating: averageRating || null
        },
        sorting: sort
      }
    })
  })
)

// POST /api/v1/vault/check-unlocks - Check and unlock eligible items
router.post(
  '/check-unlocks',
  asyncErrorHandler(async (req, res) => {
    try {
      const newlyUnlockable = await VaultItem.checkForNewUnlocks()
      const unlockedItems = []

      // Automatically unlock eligible items
      for (const { item, trigger } of newlyUnlockable) {
        try {
          await item.unlock(trigger || 'auto_unlock')
          unlockedItems.push({
            item: item.toJSON(true),
            trigger: trigger || 'auto_unlock'
          })
        } catch (error) {
          console.warn(
            `⚠️ Failed to auto-unlock ${item.itemId}:`,
            error.message
          )
        }
      }

      res.json({
        success: true,
        message:
          unlockedItems.length > 0
            ? `🎉 ${unlockedItems.length} new vault items unlocked!`
            : 'No new items available for unlock.',
        data: {
          newlyUnlocked: unlockedItems,
          count: unlockedItems.length
        }
      })
    } catch (error) {
      console.error('❌ Error checking for vault unlocks:', error)
      throw error
    }
  })
)

// ==========================================
// CATEGORY AND ORGANIZATION ENDPOINTS
// ==========================================

// GET /api/v1/vault/categories - Get vault items grouped by category
router.get(
  '/categories',
  asyncErrorHandler(async (req, res) => {
    const categories = [
      'secret_archives',
      'controversy_files',
      'beautiful_mind'
    ]
    const categoryData = {}

    for (const category of categories) {
      const items = await VaultItem.getByCategory(category)
      const unlockedCount = items.filter(item => item.isUnlocked).length

      categoryData[category] = {
        name: category,
        displayName: category
          .replace(/_/g, ' ')
          .replace(/\b\w/g, l => l.toUpperCase()),
        totalItems: items.length,
        unlockedItems: unlockedCount,
        unlockRate: items.length > 0 ? unlockedCount / items.length : 0,
        items: items.map(item => item.toJSON(item.isUnlocked))
      }
    }

    res.json({
      success: true,
      data: {
        categories: categoryData,
        summary: {
          totalCategories: categories.length,
          totalItems: Object.values(categoryData).reduce(
            (sum, cat) => sum + cat.totalItems,
            0
          ),
          totalUnlocked: Object.values(categoryData).reduce(
            (sum, cat) => sum + cat.unlockedItems,
            0
          )
        }
      }
    })
  })
)

// GET /api/v1/vault/categories/:category - Get vault items in specific category
router.get(
  '/categories/:category',
  [
    param('category').isIn([
      'secret_archives',
      'controversy_files',
      'beautiful_mind'
    ])
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { category } = req.params
    const { status, sort = 'rarity' } = req.query

    let items = await VaultItem.getByCategory(category)

    // Apply status filter
    if (status === 'unlocked') {
      items = items.filter(item => item.isUnlocked)
    } else if (status === 'locked') {
      items = items.filter(item => !item.isUnlocked)
    }

    // Sort items
    if (sort === 'rarity') {
      const rarityOrder = {
        legendary: 5,
        epic: 4,
        rare: 3,
        uncommon: 2,
        common: 1
      }
      items.sort((a, b) => rarityOrder[b.rarity] - rarityOrder[a.rarity])
    } else if (sort === 'title') {
      items.sort((a, b) => a.title.localeCompare(b.title))
    } else if (sort === 'unlock_date') {
      items.sort((a, b) => {
        if (!a.unlockedAt && !b.unlockedAt) return 0
        if (!a.unlockedAt) return 1
        if (!b.unlockedAt) return -1
        return new Date(b.unlockedAt) - new Date(a.unlockedAt)
      })
    }

    // Get category statistics
    const totalItems = items.length
    const unlockedItems = items.filter(item => item.isUnlocked).length
    const readItems = items.filter(item => item.isRead).length

    const rarityBreakdown = items.reduce((acc, item) => {
      acc[item.rarity] = (acc[item.rarity] || 0) + 1
      return acc
    }, {})

    res.json({
      success: true,
      data: {
        category,
        items: items.map(item => item.toJSON(item.isUnlocked)),
        statistics: {
          totalItems,
          unlockedItems,
          readItems,
          unlockRate: totalItems > 0 ? unlockedItems / totalItems : 0,
          readRate: unlockedItems > 0 ? readItems / unlockedItems : 0,
          rarityBreakdown
        },
        sorting: sort,
        filtering: status
      }
    })
  })
)

// GET /api/v1/vault/recent - Get recently unlocked items
router.get(
  '/recent',
  [query('limit').optional().isInt({ min: 1, max: 50 })],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { limit = 10 } = req.query

    const recentItems = await VaultItem.findByCriteria({
      isUnlocked: true,
      orderBy: 'vu.unlocked_at DESC',
      limit: parseInt(limit)
    })

    // Group by unlock date for timeline view
    const timeline = {}
    recentItems.forEach(item => {
      const unlockDate = moment(item.unlockedAt).format('YYYY-MM-DD')
      if (!timeline[unlockDate]) {
        timeline[unlockDate] = []
      }
      timeline[unlockDate].push(item.toJSON(true))
    })

    res.json({
      success: true,
      data: {
        recentItems: recentItems.map(item => item.toJSON(true)),
        timeline,
        count: recentItems.length
      }
    })
  })
)

// GET /api/v1/vault/favorites - Get user's favorite vault items
router.get(
  '/favorites',
  asyncErrorHandler(async (req, res) => {
    const favoriteItems = await VaultItem.getFavorites()

    // Group favorites by category
    const favoritesByCategory = favoriteItems.reduce((acc, item) => {
      if (!acc[item.category]) {
        acc[item.category] = []
      }
      acc[item.category].push(item.toJSON(true))
      return acc
    }, {})

    res.json({
      success: true,
      data: {
        favorites: favoriteItems.map(item => item.toJSON(true)),
        favoritesByCategory,
        count: favoriteItems.length
      }
    })
  })
)

// ==========================================
// USER INTERACTION ENDPOINTS
// ==========================================

// POST /api/v1/vault/items/:id/read - Mark vault item as read
router.post(
  '/items/:id/read',
  [param('id').notEmpty().withMessage('Item ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const vaultItem = await VaultItem.findById(id)

    if (!vaultItem) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    try {
      const result = await vaultItem.markAsRead()

      res.json({
        success: true,
        message: result.message,
        data: {
          item: vaultItem.toJSON(true),
          readAt: vaultItem.readAt
        }
      })
    } catch (error) {
      if (error.message.includes('Cannot read locked vault item')) {
        return res.status(403).json({
          success: false,
          error: {
            type: 'VAULT_ITEM_LOCKED',
            message: 'Cannot read locked vault item.'
          }
        })
      }
      throw error
    }
  })
)

// POST /api/v1/vault/items/:id/rate - Rate a vault item
router.post(
  '/items/:id/rate',
  [
    param('id').notEmpty().withMessage('Item ID is required'),
    body('rating')
      .isInt({ min: 1, max: 5 })
      .withMessage('Rating must be between 1 and 5'),
    body('notes')
      .optional()
      .isString()
      .isLength({ max: 1000 })
      .withMessage('Notes must be 1000 characters or less')
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const { rating, notes = '' } = req.body

    const vaultItem = await VaultItem.findById(id)

    if (!vaultItem) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    try {
      const result = await vaultItem.setUserRating(rating, notes)

      res.json({
        success: true,
        message: result.message,
        data: {
          item: vaultItem.toJSON(true),
          rating: vaultItem.userRating,
          notes: vaultItem.userNotes
        }
      })
    } catch (error) {
      if (error.message.includes('Cannot rate locked vault item')) {
        return res.status(403).json({
          success: false,
          error: {
            type: 'VAULT_ITEM_LOCKED',
            message: 'Cannot rate locked vault item.'
          }
        })
      }
      throw error
    }
  })
)

// POST /api/v1/vault/items/:id/favorite - Toggle favorite status
router.post(
  '/items/:id/favorite',
  [param('id').notEmpty().withMessage('Item ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const vaultItem = await VaultItem.findById(id)

    if (!vaultItem) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    try {
      const result = await vaultItem.toggleFavorite()

      res.json({
        success: true,
        message: result.message,
        data: {
          item: vaultItem.toJSON(true),
          isFavorite: result.isFavorite
        }
      })
    } catch (error) {
      if (error.message.includes('Cannot favorite locked vault item')) {
        return res.status(403).json({
          success: false,
          error: {
            type: 'VAULT_ITEM_LOCKED',
            message: 'Cannot favorite locked vault item.'
          }
        })
      }
      throw error
    }
  })
)

// ==========================================
// RECOMMENDATIONS AND DISCOVERY
// ==========================================

// GET /api/v1/vault/recommendations - Get personalized vault recommendations
router.get(
  '/recommendations',
  [query('count').optional().isInt({ min: 1, max: 20 })],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { count = 5 } = req.query

    try {
      // Get user's learning progress and preferences
      const userProgress = await getUserProgress()
      const unlockedItems = await VaultItem.getUnlockedItems()
      const favoriteItems = await VaultItem.getFavorites()

      // Analyze user preferences
      const preferredCategories = {}
      const preferredTypes = {}
      const preferredDifficulties = {}

      favoriteItems.forEach(item => {
        preferredCategories[item.category] =
          (preferredCategories[item.category] || 0) + 1
        preferredTypes[item.itemType] = (preferredTypes[item.itemType] || 0) + 1
        preferredDifficulties[item.difficulty] =
          (preferredDifficulties[item.difficulty] || 0) + 1
      })

      // Get locked items that might be unlockable soon
      const lockedItems = await VaultItem.findByCriteria({
        isUnlocked: false,
        limit: 100
      })
      const recommendations = []

      for (const item of lockedItems) {
        const unlockStatus = await item.checkUnlockConditions(userProgress)

        // Calculate recommendation score
        let score = 0

        // Close to unlocking gets higher score
        const missingCount = unlockStatus.missingRequirements.length
        if (missingCount === 0) score += 100 // Can unlock now
        else if (missingCount === 1) score += 50 // Almost there
        else if (missingCount <= 3) score += 20 // Getting close

        // Preference matching
        if (preferredCategories[item.category]) score += 10
        if (preferredTypes[item.itemType]) score += 10
        if (preferredDifficulties[item.difficulty]) score += 10

        // Rarity bonus
        const rarityBonus = {
          legendary: 20,
          epic: 15,
          rare: 10,
          uncommon: 5,
          common: 0
        }
        score += rarityBonus[item.rarity] || 0

        if (score > 0) {
          recommendations.push({
            item: item.toJSON(false),
            unlockStatus,
            recommendationScore: score,
            reason:
              missingCount === 0
                ? 'Ready to unlock!'
                : missingCount === 1
                ? 'Almost unlockable'
                : 'Matches your interests'
          })
        }
      }

      // Sort by recommendation score and limit results
      recommendations.sort(
        (a, b) => b.recommendationScore - a.recommendationScore
      )
      const topRecommendations = recommendations.slice(0, parseInt(count))

      res.json({
        success: true,
        data: {
          recommendations: topRecommendations,
          userPreferences: {
            favoriteCategories: preferredCategories,
            favoriteTypes: preferredTypes,
            favoriteDifficulties: preferredDifficulties
          },
          totalAnalyzed: lockedItems.length
        }
      })
    } catch (error) {
      console.error('❌ Error generating recommendations:', error)
      res.json({
        success: true,
        data: {
          recommendations: [],
          message: 'Unable to generate recommendations at this time.'
        }
      })
    }
  })
)

// ==========================================
// ANALYTICS AND STATISTICS
// ==========================================

// GET /api/v1/vault/statistics - Get vault statistics and progress
router.get(
  '/statistics',
  asyncErrorHandler(async (req, res) => {
    const statistics = await VaultItem.getVaultStatistics()

    // Get additional timeline data
    const unlockTimeline = await db.all(`
        SELECT 
            DATE(unlocked_at) as unlock_date,
            COUNT(*) as items_unlocked,
            category
        FROM vault_unlocks 
        WHERE unlocked_at >= date('now', '-30 days')
        GROUP BY DATE(unlocked_at), category
        ORDER BY unlock_date DESC
    `)

    // Get reading statistics
    const readingStats = await db.all(`
        SELECT 
            category,
            AVG(time_spent_reading) as avg_reading_time,
            COUNT(CASE WHEN is_read = 1 THEN 1 END) as read_count,
            COUNT(*) as total_unlocked
        FROM vault_unlocks
        GROUP BY category
    `)

    // Get rating distribution
    const ratingDistribution = await db.all(`
        SELECT 
            user_rating,
            COUNT(*) as count
        FROM vault_unlocks 
        WHERE user_rating IS NOT NULL
        GROUP BY user_rating
        ORDER BY user_rating
    `)

    res.json({
      success: true,
      data: {
        ...statistics,
        unlockTimeline,
        readingStats,
        ratingDistribution,
        insights: {
          unlockRate: statistics.unlockRate,
          readRate: statistics.readRate,
          engagementLevel:
            statistics.readRate > 0.7
              ? 'high'
              : statistics.readRate > 0.4
              ? 'medium'
              : 'low'
        }
      }
    })
  })
)

// GET /api/v1/vault/analytics - Get detailed vault analytics
router.get(
  '/analytics',
  [query('timeframe').optional().isIn(['7d', '30d', '90d', 'all'])],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { timeframe = '30d' } = req.query

    let dateFilter = ''
    if (timeframe !== 'all') {
      const days = parseInt(timeframe)
      dateFilter = `WHERE unlocked_at >= date('now', '-${days} days')`
    }

    // Unlock patterns over time
    const unlockPatterns = await db.all(`
            SELECT 
                DATE(unlocked_at) as date,
                COUNT(*) as unlocks,
                COUNT(DISTINCT category) as categories_unlocked,
                AVG(CASE 
                    WHEN vi.rarity = 'legendary' THEN 5
                    WHEN vi.rarity = 'epic' THEN 4
                    WHEN vi.rarity = 'rare' THEN 3
                    WHEN vi.rarity = 'uncommon' THEN 2
                    ELSE 1
                END) as avg_rarity_score
            FROM vault_unlocks vu
            JOIN vault_items vi ON vu.vault_item_id = vi.item_id
            ${dateFilter}
            GROUP BY DATE(unlocked_at)
            ORDER BY date DESC
        `)

    // Unlock triggers analysis
    const unlockTriggers = await db.all(`
            SELECT 
                unlock_trigger,
                COUNT(*) as count,
                COUNT(DISTINCT category) as categories
            FROM vault_unlocks 
            ${dateFilter}
            GROUP BY unlock_trigger
            ORDER BY count DESC
        `)

    // Reading behavior analysis
    const readingBehavior = await db.all(`
            SELECT 
                vi.category,
                vi.rarity,
                COUNT(*) as total_items,
                COUNT(CASE WHEN vu.is_read = 1 THEN 1 END) as read_items,
                AVG(vi.estimated_read_time) as avg_estimated_time,
                AVG(vu.time_spent_reading) as avg_actual_time
            FROM vault_items vi
            JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
            ${dateFilter}
            GROUP BY vi.category, vi.rarity
            ORDER BY vi.category, 
                CASE vi.rarity 
                    WHEN 'legendary' THEN 5
                    WHEN 'epic' THEN 4
                    WHEN 'rare' THEN 3
                    WHEN 'uncommon' THEN 2
                    ELSE 1
                END DESC
        `)

    res.json({
      success: true,
      data: {
        timeframe,
        unlockPatterns,
        unlockTriggers,
        readingBehavior,
        insights: {
          totalAnalyzed: unlockPatterns.reduce(
            (sum, day) => sum + day.unlocks,
            0
          ),
          mostCommonTrigger:
            unlockTriggers.length > 0 ? unlockTriggers[0].unlock_trigger : null,
          averageUnlocksPerDay:
            unlockPatterns.length > 0
              ? unlockPatterns.reduce((sum, day) => sum + day.unlocks, 0) /
                unlockPatterns.length
              : 0
        }
      }
    })
  })
)

// GET /api/v1/vault/search - Search vault items
router.get(
  '/search',
  [
    query('q')
      .isString()
      .isLength({ min: 1, max: 100 })
      .withMessage('Search query is required'),
    query('category')
      .optional()
      .isIn(['secret_archives', 'controversy_files', 'beautiful_mind']),
    query('status').optional().isIn(['locked', 'unlocked']),
    query('limit').optional().isInt({ min: 1, max: 50 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { q, category, status, limit = 20 } = req.query

    const criteria = { limit: parseInt(limit) }
    if (category) criteria.category = category
    if (status === 'unlocked') criteria.isUnlocked = true
    else if (status === 'locked') criteria.isUnlocked = false

    const allItems = await VaultItem.findByCriteria(criteria)

    // Search implementation
    const searchTerms = q.toLowerCase().split(' ')
    const searchResults = allItems.filter(item => {
      const searchableText = [
        item.title,
        item.description,
        item.contentPreview,
        ...item.tags,
        ...item.relatedConcepts
      ]
        .join(' ')
        .toLowerCase()

      return searchTerms.every(term => searchableText.includes(term))
    })

    // Score and sort results by relevance
    const scoredResults = searchResults.map(item => {
      let score = 0
      const titleMatch = item.title.toLowerCase().includes(q.toLowerCase())
      const descMatch = item.description.toLowerCase().includes(q.toLowerCase())

      if (titleMatch) score += 100
      if (descMatch) score += 50

      // Tag matches
      score +=
        item.tags.filter(tag => tag.toLowerCase().includes(q.toLowerCase()))
          .length * 25

      return { item, score }
    })

    scoredResults.sort((a, b) => b.score - a.score)

    res.json({
      success: true,
      data: {
        results: scoredResults.map(({ item }) => item.toJSON(item.isUnlocked)),
        query: q,
        totalResults: scoredResults.length,
        filters: { category, status }
      }
    })
  })
)

module.exports = router
