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
 * - GET /archive - Get unlocked vault items (user's collection)
 * - GET /unlock-conditions/:id - Get unlock requirements for an item
 * - POST /check-unlocks - Check and unlock eligible items based on progress
 * - GET /categories - Get vault items grouped by category
 * - GET /recent - Get recently unlocked items
 * - GET /stats - Get vault statistics and progress
 *
 * Author: Neural Explorer
 */

const express = require('express')
const { body, param, query, validationResult } = require('express-validator')
const router = express.Router()
const db = require('../config/db')
const fs = require('fs')
const path = require('path')
const {
  asyncErrorHandler,
  VaultItemLockedError
} = require('../middleware/errorHandler')

// Load vault items configuration
const VAULT_ITEMS_PATH = path.join(__dirname, '../../data/vault-items.json')

let vaultItemsConfig = {}
try {
  if (fs.existsSync(VAULT_ITEMS_PATH)) {
    vaultItemsConfig = JSON.parse(fs.readFileSync(VAULT_ITEMS_PATH, 'utf8'))
  }
} catch (error) {
  console.warn('⚠️ Could not load vault items configuration:', error.message)
}

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
 * Check if a vault item's unlock conditions are met
 * @param {Object} item - Vault item from database
 * @param {Object} userProgress - User's learning progress
 * @returns {Object} - Unlock status and missing requirements
 */
async function checkUnlockConditions (item, userProgress = null) {
  try {
    const conditions = JSON.parse(item.unlock_conditions || '{}')
    const result = {
      canUnlock: true,
      missingRequirements: [],
      progress: {}
    }

    // Get user progress if not provided
    if (!userProgress) {
      userProgress = await getUserProgress()
    }

    // Check phase completion requirements
    if (conditions.phases_completed && conditions.phases_completed.length > 0) {
      for (const requiredPhase of conditions.phases_completed) {
        const phaseStats = await db.get(
          `
                    SELECT 
                        COUNT(*) as total_lessons,
                        COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons
                    FROM learning_progress
                    WHERE phase = ?
                `,
          [requiredPhase]
        )

        const phaseComplete =
          phaseStats.total_lessons > 0 &&
          phaseStats.completed_lessons === phaseStats.total_lessons

        result.progress[`phase_${requiredPhase}`] = {
          completed: phaseStats.completed_lessons,
          total: phaseStats.total_lessons,
          isComplete: phaseComplete
        }

        if (!phaseComplete) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Complete all lessons in Phase ${requiredPhase}`
          )
        }
      }
    }

    // Check specific quest completions
    if (conditions.quests_completed && conditions.quests_completed.length > 0) {
      for (const questId of conditions.quests_completed) {
        const quest = await db.get(
          `
                    SELECT status FROM quest_completions 
                    WHERE quest_id = ? AND status IN ('completed', 'mastered')
                `,
          [questId]
        )

        result.progress[`quest_${questId}`] = {
          isComplete: !!quest
        }

        if (!quest) {
          result.canUnlock = false
          result.missingRequirements.push(`Complete quest: ${questId}`)
        }
      }
    }

    // Check minimum understanding scores
    if (conditions.min_understanding_score) {
      const avgUnderstanding = await db.get(`
                SELECT AVG(understanding_score) as avg_score
                FROM learning_progress
                WHERE status IN ('completed', 'mastered') AND understanding_score IS NOT NULL
            `)

      const currentScore = avgUnderstanding.avg_score || 0
      result.progress.understanding_score = {
        current: Math.round(currentScore * 100) / 100,
        required: conditions.min_understanding_score,
        isComplete: currentScore >= conditions.min_understanding_score
      }

      if (currentScore < conditions.min_understanding_score) {
        result.canUnlock = false
        result.missingRequirements.push(
          `Achieve minimum understanding score of ${
            conditions.min_understanding_score
          } (current: ${Math.round(currentScore * 100) / 100})`
        )
      }
    }

    // Check study time requirements
    if (conditions.min_study_time_hours) {
      const profile = await db.get(
        'SELECT total_study_time_minutes FROM user_profile WHERE id = 1'
      )
      const studyTimeHours = (profile?.total_study_time_minutes || 0) / 60

      result.progress.study_time = {
        current: Math.round(studyTimeHours * 100) / 100,
        required: conditions.min_study_time_hours,
        isComplete: studyTimeHours >= conditions.min_study_time_hours
      }

      if (studyTimeHours < conditions.min_study_time_hours) {
        result.canUnlock = false
        result.missingRequirements.push(
          `Study for at least ${
            conditions.min_study_time_hours
          } hours (current: ${Math.round(studyTimeHours * 100) / 100})`
        )
      }
    }

    // Check streak requirements
    if (conditions.min_streak_days) {
      const profile = await db.get(
        'SELECT streak_days FROM user_profile WHERE id = 1'
      )
      const currentStreak = profile?.streak_days || 0

      result.progress.streak = {
        current: currentStreak,
        required: conditions.min_streak_days,
        isComplete: currentStreak >= conditions.min_streak_days
      }

      if (currentStreak < conditions.min_streak_days) {
        result.canUnlock = false
        result.missingRequirements.push(
          `Maintain a study streak of ${conditions.min_streak_days} days (current: ${currentStreak})`
        )
      }
    }

    // Check mastery requirements
    if (conditions.lessons_mastered) {
      const masteredCount = await db.get(`
                SELECT COUNT(*) as count FROM learning_progress WHERE status = 'mastered'
            `)

      result.progress.mastery = {
        current: masteredCount.count,
        required: conditions.lessons_mastered,
        isComplete: masteredCount.count >= conditions.lessons_mastered
      }

      if (masteredCount.count < conditions.lessons_mastered) {
        result.canUnlock = false
        result.missingRequirements.push(
          `Master ${conditions.lessons_mastered} lessons (current: ${masteredCount.count})`
        )
      }
    }

    return result
  } catch (error) {
    console.error('❌ Error checking unlock conditions:', error)
    return {
      canUnlock: false,
      missingRequirements: ['Error checking unlock conditions'],
      progress: {}
    }
  }
}

/**
 * Get user's overall progress summary
 */
async function getUserProgress () {
  const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')
  const overallStats = await db.get(`
        SELECT 
            COUNT(*) as total_lessons,
            COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
            COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
            AVG(CASE WHEN status IN ('completed', 'mastered') THEN understanding_score END) as avg_understanding
        FROM learning_progress
    `)

  return {
    profile,
    stats: overallStats
  }
}

// GET /api/v1/vault/items - Get all vault items with unlock status
router.get(
  '/items',
  [
    query('category').optional().isString(),
    query('type')
      .optional()
      .isIn(['secret', 'insight', 'tool', 'story', 'challenge']),
    query('unlocked').optional().isBoolean(),
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { category, type, unlocked, limit = 50, offset = 0 } = req.query

    let whereClause = []
    let params = []

    if (category) {
      whereClause.push('category = ?')
      params.push(category)
    }

    if (type) {
      whereClause.push('item_type = ?')
      params.push(type)
    }

    if (unlocked !== undefined) {
      whereClause.push('is_unlocked = ?')
      params.push(unlocked ? 1 : 0)
    }

    const whereSQL =
      whereClause.length > 0 ? `WHERE ${whereClause.join(' AND ')}` : ''

    // Get total count
    const countQuery = `SELECT COUNT(*) as total FROM vault_items ${whereSQL}`
    const { total } = await db.get(countQuery, params)

    // Get vault items with pagination
    const itemsQuery = `
        SELECT 
            id, item_id, title, description, category, item_type,
            rarity, unlock_conditions, content_preview, content_full,
            is_unlocked, unlocked_at, created_at, updated_at
        FROM vault_items
        ${whereSQL}
        ORDER BY rarity DESC, created_at DESC
        LIMIT ? OFFSET ?
    `

    const items = await db.all(itemsQuery, [...params, limit, offset])

    // Check unlock status for each item and prepare response
    const itemsWithStatus = await Promise.all(
      items.map(async item => {
        let unlockStatus = null

        if (!item.is_unlocked) {
          unlockStatus = await checkUnlockConditions(item)
        }

        return {
          id: item.item_id,
          title: item.title,
          description: item.description,
          category: item.category,
          type: item.item_type,
          rarity: item.rarity,
          isUnlocked: !!item.is_unlocked,
          unlockedAt: item.unlocked_at,
          contentPreview: item.content_preview,
          contentFull: item.is_unlocked ? item.content_full : null,
          unlockStatus: unlockStatus,
          createdAt: item.created_at
        }
      })
    )

    res.json({
      success: true,
      data: {
        items: itemsWithStatus,
        pagination: {
          total,
          limit: parseInt(limit),
          offset: parseInt(offset),
          hasMore: parseInt(offset) + parseInt(limit) < total
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

    const item = await db.get(
      `
        SELECT 
            id, item_id, title, description, category, item_type,
            rarity, unlock_conditions, content_preview, content_full,
            is_unlocked, unlocked_at, created_at, updated_at
        FROM vault_items
        WHERE item_id = ?
    `,
      [id]
    )

    if (!item) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    // Check unlock status
    let unlockStatus = null
    if (!item.is_unlocked) {
      unlockStatus = await checkUnlockConditions(item)
    }

    res.json({
      success: true,
      data: {
        item: {
          id: item.item_id,
          title: item.title,
          description: item.description,
          category: item.category,
          type: item.item_type,
          rarity: item.rarity,
          isUnlocked: !!item.is_unlocked,
          unlockedAt: item.unlocked_at,
          contentPreview: item.content_preview,
          contentFull: item.is_unlocked ? item.content_full : null,
          unlockStatus: unlockStatus,
          createdAt: item.created_at,
          updatedAt: item.updated_at
        }
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

    // Get vault item
    const item = await db.get(
      `
        SELECT * FROM vault_items WHERE item_id = ?
    `,
      [id]
    )

    if (!item) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    // Check if already unlocked
    if (item.is_unlocked) {
      return res.json({
        success: true,
        message: 'Item is already unlocked!',
        data: {
          itemId: id,
          title: item.title,
          alreadyUnlocked: true
        }
      })
    }

    // Check unlock conditions
    const unlockStatus = await checkUnlockConditions(item)

    if (!unlockStatus.canUnlock) {
      throw new VaultItemLockedError(id, unlockStatus.missingRequirements)
    }

    // Unlock the item
    await db.run(
      `
        UPDATE vault_items
        SET is_unlocked = 1,
            unlocked_at = datetime('now'),
            updated_at = datetime('now')
        WHERE item_id = ?
    `,
      [id]
    )

    // Record unlock event
    await db.run(
      `
        INSERT INTO vault_unlock_events (
            item_id, unlock_conditions_met, unlocked_at, created_at
        ) VALUES (?, ?, datetime('now'), datetime('now'))
    `,
      [id, JSON.stringify(unlockStatus.progress)]
    )

    res.json({
      success: true,
      message: 'Vault item unlocked successfully! 🎉',
      data: {
        itemId: id,
        title: item.title,
        category: item.category,
        type: item.item_type,
        rarity: item.rarity,
        contentFull: item.content_full,
        unlockedAt: new Date().toISOString()
      }
    })
  })
)

// GET /api/v1/vault/archive - Get unlocked vault items (user's collection)
router.get(
  '/archive',
  [
    query('category').optional().isString(),
    query('type')
      .optional()
      .isIn(['secret', 'insight', 'tool', 'story', 'challenge']),
    query('rarity')
      .optional()
      .isIn(['common', 'uncommon', 'rare', 'epic', 'legendary']),
    query('sortBy').optional().isIn(['unlocked_at', 'rarity', 'title']),
    query('sortOrder').optional().isIn(['asc', 'desc']),
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      category,
      type,
      rarity,
      sortBy = 'unlocked_at',
      sortOrder = 'desc',
      limit = 50,
      offset = 0
    } = req.query

    let whereClause = ['is_unlocked = 1']
    let params = []

    if (category) {
      whereClause.push('category = ?')
      params.push(category)
    }

    if (type) {
      whereClause.push('item_type = ?')
      params.push(type)
    }

    if (rarity) {
      whereClause.push('rarity = ?')
      params.push(rarity)
    }

    const whereSQL = `WHERE ${whereClause.join(' AND ')}`

    // Build ORDER BY clause
    let orderByClause = ''
    switch (sortBy) {
      case 'rarity':
        orderByClause = `ORDER BY 
                CASE rarity 
                    WHEN 'legendary' THEN 5
                    WHEN 'epic' THEN 4
                    WHEN 'rare' THEN 3
                    WHEN 'uncommon' THEN 2
                    ELSE 1
                END ${sortOrder.toUpperCase()}`
        break
      case 'title':
        orderByClause = `ORDER BY title ${sortOrder.toUpperCase()}`
        break
      default:
        orderByClause = `ORDER BY unlocked_at ${sortOrder.toUpperCase()}`
    }

    // Get total count
    const countQuery = `SELECT COUNT(*) as total FROM vault_items ${whereSQL}`
    const { total } = await db.get(countQuery, params)

    // Get unlocked items
    const itemsQuery = `
        SELECT 
            item_id, title, description, category, item_type, rarity,
            content_preview, content_full, unlocked_at, created_at
        FROM vault_items
        ${whereSQL}
        ${orderByClause}
        LIMIT ? OFFSET ?
    `

    const items = await db.all(itemsQuery, [...params, limit, offset])

    // Group items by category for better organization
    const itemsByCategory = {}
    items.forEach(item => {
      if (!itemsByCategory[item.category]) {
        itemsByCategory[item.category] = []
      }
      itemsByCategory[item.category].push({
        id: item.item_id,
        title: item.title,
        description: item.description,
        type: item.item_type,
        rarity: item.rarity,
        contentPreview: item.content_preview,
        contentFull: item.content_full,
        unlockedAt: item.unlocked_at,
        createdAt: item.created_at
      })
    })

    res.json({
      success: true,
      data: {
        items: items.map(item => ({
          id: item.item_id,
          title: item.title,
          description: item.description,
          category: item.category,
          type: item.item_type,
          rarity: item.rarity,
          contentPreview: item.content_preview,
          contentFull: item.content_full,
          unlockedAt: item.unlocked_at,
          createdAt: item.created_at
        })),
        itemsByCategory,
        pagination: {
          total,
          limit: parseInt(limit),
          offset: parseInt(offset),
          hasMore: parseInt(offset) + parseInt(limit) < total
        }
      }
    })
  })
)

// GET /api/v1/vault/unlock-conditions/:id - Get unlock requirements for an item
router.get(
  '/unlock-conditions/:id',
  [param('id').notEmpty().withMessage('Item ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const item = await db.get(
      `
        SELECT item_id, title, unlock_conditions, is_unlocked
        FROM vault_items
        WHERE item_id = ?
    `,
      [id]
    )

    if (!item) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'VAULT_ITEM_NOT_FOUND',
          message: 'Vault item not found.'
        }
      })
    }

    if (item.is_unlocked) {
      return res.json({
        success: true,
        data: {
          itemId: id,
          title: item.title,
          isUnlocked: true,
          message: 'This item is already unlocked!'
        }
      })
    }

    // Get detailed unlock status
    const unlockStatus = await checkUnlockConditions(item)

    res.json({
      success: true,
      data: {
        itemId: id,
        title: item.title,
        isUnlocked: false,
        canUnlock: unlockStatus.canUnlock,
        requirements: unlockStatus.missingRequirements,
        progress: unlockStatus.progress,
        completionPercentage:
          unlockStatus.missingRequirements.length === 0
            ? 100
            : Math.round(
                (Object.keys(unlockStatus.progress).filter(
                  key => unlockStatus.progress[key].isComplete !== false
                ).length /
                  Object.keys(unlockStatus.progress).length) *
                  100
              )
      }
    })
  })
)

// POST /api/v1/vault/check-unlocks - Check and unlock eligible items based on progress
router.post(
  '/check-unlocks',
  asyncErrorHandler(async (req, res) => {
    // Get all locked vault items
    const lockedItems = await db.all(`
        SELECT item_id, title, unlock_conditions
        FROM vault_items
        WHERE is_unlocked = 0
    `)

    const unlockedItems = []
    const userProgress = await getUserProgress()

    // Check each locked item
    for (const item of lockedItems) {
      const unlockStatus = await checkUnlockConditions(item, userProgress)

      if (unlockStatus.canUnlock) {
        // Unlock the item
        await db.run(
          `
                UPDATE vault_items
                SET is_unlocked = 1,
                    unlocked_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE item_id = ?
            `,
          [item.item_id]
        )

        // Record unlock event
        await db.run(
          `
                INSERT INTO vault_unlock_events (
                    item_id, unlock_conditions_met, unlocked_at, created_at
                ) VALUES (?, ?, datetime('now'), datetime('now'))
            `,
          [item.item_id, JSON.stringify(unlockStatus.progress)]
        )

        unlockedItems.push({
          id: item.item_id,
          title: item.title
        })
      }
    }

    res.json({
      success: true,
      message:
        unlockedItems.length > 0
          ? `${unlockedItems.length} new vault items unlocked!`
          : 'No new items unlocked at this time.',
      data: {
        newlyUnlocked: unlockedItems,
        count: unlockedItems.length
      }
    })
  })
)

// GET /api/v1/vault/categories - Get vault items grouped by category
router.get(
  '/categories',
  asyncErrorHandler(async (req, res) => {
    const categories = await db.all(`
        SELECT 
            category,
            COUNT(*) as total_items,
            COUNT(CASE WHEN is_unlocked = 1 THEN 1 END) as unlocked_items,
            GROUP_CONCAT(CASE WHEN is_unlocked = 1 THEN item_type END) as unlocked_types
        FROM vault_items
        GROUP BY category
        ORDER BY category
    `)

    const categoryData = categories.map(cat => ({
      name: cat.category,
      totalItems: cat.total_items,
      unlockedItems: cat.unlocked_items,
      completionPercentage: Math.round(
        (cat.unlocked_items / cat.total_items) * 100
      ),
      unlockedTypes: cat.unlocked_types ? cat.unlocked_types.split(',') : []
    }))

    res.json({
      success: true,
      data: {
        categories: categoryData,
        totalCategories: categories.length,
        overallStats: {
          totalItems: categories.reduce((sum, cat) => sum + cat.total_items, 0),
          totalUnlocked: categories.reduce(
            (sum, cat) => sum + cat.unlocked_items,
            0
          )
        }
      }
    })
  })
)

// GET /api/v1/vault/recent - Get recently unlocked items
router.get(
  '/recent',
  [
    query('limit').optional().isInt({ min: 1, max: 50 }),
    query('days').optional().isInt({ min: 1, max: 365 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { limit = 10, days = 30 } = req.query

    const cutoffDate = new Date()
    cutoffDate.setDate(cutoffDate.getDate() - days)

    const recentItems = await db.all(
      `
        SELECT 
            item_id, title, description, category, item_type, rarity,
            content_preview, unlocked_at
        FROM vault_items
        WHERE is_unlocked = 1 AND unlocked_at >= ?
        ORDER BY unlocked_at DESC
        LIMIT ?
    `,
      [cutoffDate.toISOString(), limit]
    )

    res.json({
      success: true,
      data: {
        recentlyUnlocked: recentItems.map(item => ({
          id: item.item_id,
          title: item.title,
          description: item.description,
          category: item.category,
          type: item.item_type,
          rarity: item.rarity,
          contentPreview: item.content_preview,
          unlockedAt: item.unlocked_at
        })),
        count: recentItems.length,
        timeframe: `${days} days`
      }
    })
  })
)

// GET /api/v1/vault/stats - Get vault statistics and progress
router.get(
  '/stats',
  asyncErrorHandler(async (req, res) => {
    // Get overall vault statistics
    const overallStats = await db.get(`
        SELECT 
            COUNT(*) as total_items,
            COUNT(CASE WHEN is_unlocked = 1 THEN 1 END) as unlocked_items,
            COUNT(CASE WHEN is_unlocked = 0 THEN 1 END) as locked_items
        FROM vault_items
    `)

    // Get stats by category
    const categoryStats = await db.all(`
        SELECT 
            category,
            COUNT(*) as total,
            COUNT(CASE WHEN is_unlocked = 1 THEN 1 END) as unlocked
        FROM vault_items
        GROUP BY category
        ORDER BY category
    `)

    // Get stats by rarity
    const rarityStats = await db.all(`
        SELECT 
            rarity,
            COUNT(*) as total,
            COUNT(CASE WHEN is_unlocked = 1 THEN 1 END) as unlocked
        FROM vault_items
        GROUP BY rarity
        ORDER BY 
            CASE rarity 
                WHEN 'legendary' THEN 5
                WHEN 'epic' THEN 4
                WHEN 'rare' THEN 3
                WHEN 'uncommon' THEN 2
                ELSE 1
            END DESC
    `)

    // Get stats by type
    const typeStats = await db.all(`
        SELECT 
            item_type,
            COUNT(*) as total,
            COUNT(CASE WHEN is_unlocked = 1 THEN 1 END) as unlocked
        FROM vault_items
        GROUP BY item_type
        ORDER BY item_type
    `)

    // Get recent unlock activity
    const recentActivity = await db.all(`
        SELECT 
            DATE(unlocked_at) as date,
            COUNT(*) as unlocks_count
        FROM vault_items
        WHERE is_unlocked = 1 AND unlocked_at >= date('now', '-30 days')
        GROUP BY DATE(unlocked_at)
        ORDER BY date DESC
    `)

    const completionPercentage =
      overallStats.total_items > 0
        ? Math.round(
            (overallStats.unlocked_items / overallStats.total_items) * 100
          )
        : 0

    res.json({
      success: true,
      data: {
        overall: {
          totalItems: overallStats.total_items,
          unlockedItems: overallStats.unlocked_items,
          lockedItems: overallStats.locked_items,
          completionPercentage
        },
        byCategory: categoryStats.map(stat => ({
          category: stat.category,
          total: stat.total,
          unlocked: stat.unlocked,
          completionPercentage: Math.round((stat.unlocked / stat.total) * 100)
        })),
        byRarity: rarityStats.map(stat => ({
          rarity: stat.rarity,
          total: stat.total,
          unlocked: stat.unlocked,
          completionPercentage: Math.round((stat.unlocked / stat.total) * 100)
        })),
        byType: typeStats.map(stat => ({
          type: stat.item_type,
          total: stat.total,
          unlocked: stat.unlocked,
          completionPercentage: Math.round((stat.unlocked / stat.total) * 100)
        })),
        recentActivity: recentActivity.map(activity => ({
          date: activity.date,
          unlocksCount: activity.unlocks_count
        }))
      }
    })
  })
)

module.exports = router
