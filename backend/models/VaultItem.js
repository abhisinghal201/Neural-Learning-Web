/**
 * Neural Odyssey Vault Item Model
 *
 * Handles the Neural Vault system - a gamified reward mechanism that unlocks
 * special content, insights, and treasures as users progress through their
 * machine learning journey. Vault items are categorized into different types
 * and have complex unlock conditions based on learning achievements.
 *
 * Vault Categories:
 * - secret_archives: Hidden mathematical insights and historical discoveries
 * - controversy_files: Debates, failures, and controversial topics in AI/ML
 * - beautiful_mind: Inspirational stories, breakthrough moments, and creativity
 *
 * Item Types:
 * - secret: Hidden knowledge and insider insights
 * - tool: Practical utilities and advanced techniques
 * - artifact: Historical documents and breakthrough papers
 * - story: Inspirational narratives and personal journeys
 * - wisdom: Deep philosophical insights and principles
 *
 * Features:
 * - Complex unlock condition evaluation
 * - Progressive content revelation
 * - Rarity-based reward system
 * - User interaction tracking (read, favorite, rating)
 * - Content recommendation engine
 * - Achievement-based unlocking
 *
 * Author: Neural Explorer
 */

const db = require('../config/db')
const fs = require('fs')
const path = require('path')
const moment = require('moment')
const crypto = require('crypto')

class VaultItem {
  constructor (data = {}) {
    this.itemId = data.itemId || this.generateItemId()
    this.title = data.title || ''
    this.description = data.description || ''
    this.category = data.category || 'secret_archives'
    this.itemType = data.itemType || 'secret'
    this.rarity = data.rarity || 'common'
    this.unlockConditions = data.unlockConditions || {}
    this.contentPreview = data.contentPreview || ''
    this.contentFull = data.contentFull || ''
    this.tags = data.tags || []
    this.estimatedReadTime = data.estimatedReadTime || 5
    this.difficulty = data.difficulty || 'beginner'
    this.relatedConcepts = data.relatedConcepts || []
    this.sources = data.sources || []
    this.isUnlocked = data.isUnlocked || false
    this.unlockedAt = data.unlockedAt || null
    this.isRead = data.isRead || false
    this.readAt = data.readAt || null
    this.isFavorite = data.isFavorite || false
    this.userRating = data.userRating || null
    this.userNotes = data.userNotes || ''
    this.accessCount = data.accessCount || 0
    this.createdAt = data.createdAt || new Date().toISOString()
    this.updatedAt = data.updatedAt || new Date().toISOString()
  }

  /**
   * Generate unique vault item ID
   * @returns {string} - Unique vault item identifier
   */
  generateItemId () {
    const timestamp = Date.now().toString(36)
    const random = crypto.randomBytes(3).toString('hex')
    return `vault_${timestamp}_${random}`
  }

  /**
   * Validate vault item data
   * @returns {Object} - Validation result with errors if any
   */
  validate () {
    const errors = []

    if (!this.title || this.title.trim().length === 0) {
      errors.push('Vault item title is required')
    }

    if (!this.description || this.description.trim().length === 0) {
      errors.push('Vault item description is required')
    }

    if (
      !['secret_archives', 'controversy_files', 'beautiful_mind'].includes(
        this.category
      )
    ) {
      errors.push('Invalid vault category')
    }

    if (
      !['secret', 'tool', 'artifact', 'story', 'wisdom'].includes(this.itemType)
    ) {
      errors.push('Invalid item type')
    }

    if (
      !['common', 'uncommon', 'rare', 'epic', 'legendary'].includes(this.rarity)
    ) {
      errors.push('Invalid rarity level')
    }

    if (
      !['beginner', 'intermediate', 'advanced', 'expert'].includes(
        this.difficulty
      )
    ) {
      errors.push('Invalid difficulty level')
    }

    if (!this.contentPreview || this.contentPreview.trim().length === 0) {
      errors.push('Content preview is required')
    }

    if (!this.contentFull || this.contentFull.trim().length === 0) {
      errors.push('Full content is required')
    }

    if (
      this.userRating !== null &&
      (!Number.isInteger(this.userRating) ||
        this.userRating < 1 ||
        this.userRating > 5)
    ) {
      errors.push('User rating must be an integer between 1 and 5')
    }

    return {
      isValid: errors.length === 0,
      errors
    }
  }

  /**
   * Check if vault item can be unlocked based on user progress
   * @param {Object} userProgress - User's learning progress data
   * @returns {Promise<Object>} - Unlock status and requirements
   */
  async checkUnlockConditions (userProgress = null) {
    try {
      const conditions = this.unlockConditions
      const result = {
        canUnlock: true,
        missingRequirements: [],
        progress: {},
        unlockTrigger: null
      }

      // Get user progress if not provided
      if (!userProgress) {
        userProgress = await this.getUserProgress()
      }

      // Check phases completed requirement
      if (conditions.phasesCompleted && conditions.phasesCompleted.length > 0) {
        for (const requiredPhase of conditions.phasesCompleted) {
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

          const completionRate =
            phaseStats.total_lessons > 0
              ? phaseStats.completed_lessons / phaseStats.total_lessons
              : 0

          if (completionRate < 0.8) {
            // Require 80% completion
            result.canUnlock = false
            result.missingRequirements.push(
              `Complete 80% of Phase ${requiredPhase} lessons`
            )
          } else {
            result.unlockTrigger = `phase_${requiredPhase}_completion`
          }

          result.progress[`phase_${requiredPhase}_completion`] = completionRate
        }
      }

      // Check minimum understanding score
      if (conditions.minUnderstandingScore) {
        const avgScore = await db.get(`
                    SELECT AVG(mastery_score) as avg_score
                    FROM learning_progress 
                    WHERE mastery_score IS NOT NULL
                        AND status IN ('completed', 'mastered')
                `)

        const currentScore = avgScore.avg_score || 0
        if (currentScore < conditions.minUnderstandingScore) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Achieve average understanding score of ${(
              conditions.minUnderstandingScore * 100
            ).toFixed(0)}%`
          )
        }

        result.progress.averageUnderstanding = currentScore
      }

      // Check total study time requirement
      if (conditions.minStudyTimeHours) {
        const profile = await db.get(
          'SELECT total_study_minutes FROM user_profile WHERE id = 1'
        )
        const currentHours = profile ? profile.total_study_minutes / 60 : 0

        if (currentHours < conditions.minStudyTimeHours) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Study for ${(conditions.minStudyTimeHours - currentHours).toFixed(
              1
            )} more hours`
          )
        }

        result.progress.studyHours = currentHours
        if (currentHours >= conditions.minStudyTimeHours) {
          result.unlockTrigger = 'study_time_milestone'
        }
      }

      // Check quest completion requirements
      if (conditions.requiredQuests && conditions.requiredQuests.length > 0) {
        for (const questId of conditions.requiredQuests) {
          const quest = await db.get(
            `
                        SELECT * FROM quest_completions 
                        WHERE quest_id = ? AND status IN ('completed', 'mastered')
                    `,
            [questId]
          )

          if (!quest) {
            result.canUnlock = false
            result.missingRequirements.push(`Complete quest: ${questId}`)
          } else {
            result.unlockTrigger = `quest_${questId}_completion`
          }
        }
      }

      // Check streak requirements
      if (conditions.minStreakDays) {
        const profile = await db.get(
          'SELECT current_streak_days FROM user_profile WHERE id = 1'
        )
        const currentStreak = profile ? profile.current_streak_days : 0

        if (currentStreak < conditions.minStreakDays) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Maintain a learning streak of ${conditions.minStreakDays} days`
          )
        }

        result.progress.currentStreak = currentStreak
        if (currentStreak >= conditions.minStreakDays) {
          result.unlockTrigger = 'streak_milestone'
        }
      }

      // Check skill points requirements
      if (conditions.skillPointsRequired) {
        for (const [category, requiredPoints] of Object.entries(
          conditions.skillPointsRequired
        )) {
          const skillPoints = await db.get(
            `
                        SELECT SUM(points_earned) as total_points
                        FROM skill_points 
                        WHERE category = ?
                    `,
            [category]
          )

          const currentPoints = skillPoints.total_points || 0
          if (currentPoints < requiredPoints) {
            result.canUnlock = false
            result.missingRequirements.push(
              `Earn ${
                requiredPoints - currentPoints
              } more ${category} skill points`
            )
          }

          result.progress[`${category}_points`] = currentPoints
        }
      }

      // Check specific lesson mastery
      if (conditions.masteredLessons && conditions.masteredLessons.length > 0) {
        for (const lessonId of conditions.masteredLessons) {
          const lesson = await db.get(
            `
                        SELECT * FROM learning_progress 
                        WHERE lesson_id = ? AND status = 'mastered'
                    `,
            [lessonId]
          )

          if (!lesson) {
            result.canUnlock = false
            result.missingRequirements.push(`Master lesson: ${lessonId}`)
          } else {
            result.unlockTrigger = `lesson_${lessonId}_mastery`
          }
        }
      }

      // Check concept connections (knowledge graph depth)
      if (conditions.minConceptConnections) {
        const connections = await db.get(`
                    SELECT COUNT(*) as connection_count
                    FROM knowledge_connections 
                    WHERE strength >= 0.7
                `)

        const currentConnections = connections.connection_count || 0
        if (currentConnections < conditions.minConceptConnections) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Build ${
              conditions.minConceptConnections - currentConnections
            } more strong concept connections`
          )
        }

        result.progress.conceptConnections = currentConnections
      }

      // Check previous vault unlocks (prerequisite vault items)
      if (
        conditions.requiredVaultItems &&
        conditions.requiredVaultItems.length > 0
      ) {
        for (const vaultItemId of conditions.requiredVaultItems) {
          const vaultItem = await db.get(
            `
                        SELECT * FROM vault_unlocks 
                        WHERE vault_item_id = ?
                    `,
            [vaultItemId]
          )

          if (!vaultItem) {
            result.canUnlock = false
            result.missingRequirements.push(`Unlock vault item: ${vaultItemId}`)
          }
        }
      }

      // Special time-based conditions
      if (conditions.minDaysActive) {
        const profile = await db.get(
          'SELECT created_at FROM user_profile WHERE id = 1'
        )
        if (profile) {
          const daysActive = moment().diff(moment(profile.created_at), 'days')
          if (daysActive < conditions.minDaysActive) {
            result.canUnlock = false
            result.missingRequirements.push(
              `Continue learning for ${
                conditions.minDaysActive - daysActive
              } more days`
            )
          }

          result.progress.daysActive = daysActive
        }
      }

      return result
    } catch (error) {
      console.error('❌ Error checking vault unlock conditions:', error)
      return {
        canUnlock: false,
        missingRequirements: ['Error checking requirements'],
        progress: {},
        unlockTrigger: null
      }
    }
  }

  /**
   * Get user's overall progress data
   * @returns {Promise<Object>} - User progress summary
   */
  async getUserProgress () {
    try {
      const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')
      const lessonProgress = await db.all('SELECT * FROM learning_progress')
      const questProgress = await db.all('SELECT * FROM quest_completions')
      const skillPoints = await db.all('SELECT * FROM skill_points')
      const vaultUnlocks = await db.all('SELECT * FROM vault_unlocks')
      const knowledgeGraph = await db.all('SELECT * FROM knowledge_connections')

      return {
        profile,
        lessons: lessonProgress,
        quests: questProgress,
        skills: skillPoints,
        vault: vaultUnlocks,
        knowledge: knowledgeGraph
      }
    } catch (error) {
      console.error('❌ Error getting user progress:', error)
      return null
    }
  }

  /**
   * Unlock this vault item
   * @param {string} trigger - What triggered the unlock
   * @returns {Promise<Object>} - Unlock result
   */
  async unlock (trigger = 'manual') {
    try {
      const unlockStatus = await this.checkUnlockConditions()

      if (!unlockStatus.canUnlock) {
        throw new Error(
          `Cannot unlock vault item: ${unlockStatus.missingRequirements.join(
            ', '
          )}`
        )
      }

      // Record unlock in vault_unlocks table
      await db.run(
        `
                INSERT OR REPLACE INTO vault_unlocks (
                    vault_item_id, category, unlocked_at, unlock_trigger,
                    unlock_phase, unlock_week, unlock_lesson_id
                ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
            `,
        [
          this.itemId,
          this.category,
          trigger,
          null, // Will be set based on current progress
          null,
          null
        ]
      )

      // Update vault_items table
      await db.run(
        `
                UPDATE vault_items 
                SET is_unlocked = 1,
                    unlocked_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE item_id = ?
            `,
        [this.itemId]
      )

      // Record unlock event for analytics
      await db.run(
        `
                INSERT INTO vault_unlock_events (
                    item_id, unlock_conditions_met, unlocked_at, created_at
                ) VALUES (?, ?, datetime('now'), datetime('now'))
            `,
        [this.itemId, JSON.stringify(unlockStatus.progress)]
      )

      // Award bonus skill points for vault unlocks
      const bonusPoints = this.getRarityBonus()
      if (bonusPoints > 0) {
        await db.run(
          `
                    INSERT INTO skill_points (
                        category, points_earned, reason, source_type,
                        related_quest_id, earned_at
                    ) VALUES ('exploration', ?, ?, 'vault_unlock', ?, datetime('now'))
                `,
          [
            bonusPoints,
            `Unlocked ${this.rarity} vault item: ${this.title}`,
            this.itemId
          ]
        )
      }

      this.isUnlocked = true
      this.unlockedAt = new Date().toISOString()

      return {
        success: true,
        message: `🎉 Vault item "${this.title}" unlocked!`,
        bonusPoints,
        trigger
      }
    } catch (error) {
      console.error('❌ Error unlocking vault item:', error)
      throw error
    }
  }

  /**
   * Mark vault item as read
   * @returns {Promise<Object>} - Read result
   */
  async markAsRead () {
    try {
      if (!this.isUnlocked) {
        throw new Error('Cannot read locked vault item')
      }

      await db.run(
        `
                UPDATE vault_unlocks 
                SET is_read = 1,
                    read_at = datetime('now'),
                    time_spent_reading = COALESCE(time_spent_reading, 0) + 1
                WHERE vault_item_id = ?
            `,
        [this.itemId]
      )

      await db.run(
        `
                UPDATE vault_items 
                SET access_count = access_count + 1,
                    updated_at = datetime('now')
                WHERE item_id = ?
            `,
        [this.itemId]
      )

      this.isRead = true
      this.readAt = new Date().toISOString()
      this.accessCount += 1

      return {
        success: true,
        message: 'Vault item marked as read'
      }
    } catch (error) {
      console.error('❌ Error marking vault item as read:', error)
      throw error
    }
  }

  /**
   * Set user rating for vault item
   * @param {number} rating - Rating (1-5)
   * @param {string} notes - Optional user notes
   * @returns {Promise<Object>} - Rating result
   */
  async setUserRating (rating, notes = '') {
    try {
      if (!this.isUnlocked) {
        throw new Error('Cannot rate locked vault item')
      }

      if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
        throw new Error('Rating must be an integer between 1 and 5')
      }

      await db.run(
        `
                UPDATE vault_unlocks 
                SET user_rating = ?,
                    user_notes = ?
                WHERE vault_item_id = ?
            `,
        [rating, notes, this.itemId]
      )

      this.userRating = rating
      this.userNotes = notes

      return {
        success: true,
        message: 'Rating saved successfully'
      }
    } catch (error) {
      console.error('❌ Error setting user rating:', error)
      throw error
    }
  }

  /**
   * Toggle favorite status
   * @returns {Promise<Object>} - Favorite toggle result
   */
  async toggleFavorite () {
    try {
      if (!this.isUnlocked) {
        throw new Error('Cannot favorite locked vault item')
      }

      const newFavoriteStatus = !this.isFavorite

      await db.run(
        `
                UPDATE vault_unlocks 
                SET is_favorite = ?
                WHERE vault_item_id = ?
            `,
        [newFavoriteStatus ? 1 : 0, this.itemId]
      )

      this.isFavorite = newFavoriteStatus

      return {
        success: true,
        isFavorite: newFavoriteStatus,
        message: newFavoriteStatus
          ? 'Added to favorites'
          : 'Removed from favorites'
      }
    } catch (error) {
      console.error('❌ Error toggling favorite:', error)
      throw error
    }
  }

  /**
   * Get rarity-based bonus points
   * @returns {number} - Bonus points for unlocking
   */
  getRarityBonus () {
    const bonuses = {
      common: 5,
      uncommon: 10,
      rare: 20,
      epic: 35,
      legendary: 50
    }

    return bonuses[this.rarity] || 5
  }

  /**
   * Get rarity color for UI
   * @returns {string} - CSS color class
   */
  getRarityColor () {
    const colors = {
      common: 'text-gray-400',
      uncommon: 'text-green-400',
      rare: 'text-blue-400',
      epic: 'text-purple-400',
      legendary: 'text-yellow-400'
    }

    return colors[this.rarity] || 'text-gray-400'
  }

  /**
   * Get content with appropriate access level
   * @param {boolean} forcePreview - Force preview even if unlocked
   * @returns {string} - Content to display
   */
  getContent (forcePreview = false) {
    if (this.isUnlocked && !forcePreview) {
      return this.contentFull
    }
    return this.contentPreview
  }

  /**
   * Get related vault items based on tags and concepts
   * @returns {Promise<Array>} - Array of related vault items
   */
  async getRelatedItems () {
    try {
      // Find items with overlapping tags
      const tagQuery = this.tags.map(() => '?').join(',')
      const related = await db.all(
        `
                SELECT DISTINCT vi.*, vu.is_unlocked
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                WHERE vi.item_id != ?
                    AND (vi.category = ? OR 
                         EXISTS (
                             SELECT 1 FROM json_each(vi.tags) 
                             WHERE value IN (${tagQuery})
                         ))
                ORDER BY CASE WHEN vu.is_unlocked THEN 1 ELSE 0 END DESC,
                         vi.rarity DESC
                LIMIT 5
            `,
        [this.itemId, this.category, ...this.tags]
      )

      return related.map(item => VaultItem.fromDatabase(item))
    } catch (error) {
      console.error('❌ Error getting related vault items:', error)
      return []
    }
  }

  /**
   * Get vault item statistics
   * @returns {Promise<Object>} - Item statistics
   */
  async getStatistics () {
    try {
      const stats = await db.get(
        `
                SELECT 
                    vu.time_spent_reading,
                    vu.user_rating,
                    vi.access_count,
                    vu.unlocked_at,
                    vu.read_at
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                WHERE vi.item_id = ?
            `,
        [this.itemId]
      )

      return {
        timeSpentReading: stats?.time_spent_reading || 0,
        userRating: stats?.user_rating || null,
        accessCount: stats?.access_count || 0,
        unlockedAt: stats?.unlocked_at || null,
        readAt: stats?.read_at || null,
        estimatedReadTime: this.estimatedReadTime,
        rarity: this.rarity,
        category: this.category
      }
    } catch (error) {
      console.error('❌ Error getting vault item statistics:', error)
      return {
        timeSpentReading: 0,
        userRating: null,
        accessCount: 0,
        unlockedAt: null,
        readAt: null,
        estimatedReadTime: this.estimatedReadTime,
        rarity: this.rarity,
        category: this.category
      }
    }
  }

  /**
   * Save vault item to database
   * @returns {Promise<Object>} - Save result
   */
  async save () {
    try {
      const validation = this.validate()
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`)
      }

      const exists = await db.get(
        'SELECT item_id FROM vault_items WHERE item_id = ?',
        [this.itemId]
      )

      if (exists) {
        // Update existing vault item
        await db.run(
          `
                    UPDATE vault_items SET
                        title = ?, description = ?, category = ?, item_type = ?,
                        rarity = ?, unlock_conditions = ?, content_preview = ?,
                        content_full = ?, tags = ?, estimated_read_time = ?,
                        difficulty = ?, related_concepts = ?, sources = ?,
                        updated_at = datetime('now')
                    WHERE item_id = ?
                `,
          [
            this.title,
            this.description,
            this.category,
            this.itemType,
            this.rarity,
            JSON.stringify(this.unlockConditions),
            this.contentPreview,
            this.contentFull,
            JSON.stringify(this.tags),
            this.estimatedReadTime,
            this.difficulty,
            JSON.stringify(this.relatedConcepts),
            JSON.stringify(this.sources),
            this.itemId
          ]
        )
      } else {
        // Create new vault item
        await db.run(
          `
                    INSERT INTO vault_items (
                        item_id, title, description, category, item_type, rarity,
                        unlock_conditions, content_preview, content_full, tags,
                        estimated_read_time, difficulty, related_concepts, sources,
                        is_unlocked, unlocked_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                             datetime('now'), datetime('now'))
                `,
          [
            this.itemId,
            this.title,
            this.description,
            this.category,
            this.itemType,
            this.rarity,
            JSON.stringify(this.unlockConditions),
            this.contentPreview,
            this.contentFull,
            JSON.stringify(this.tags),
            this.estimatedReadTime,
            this.difficulty,
            JSON.stringify(this.relatedConcepts),
            JSON.stringify(this.sources),
            this.isUnlocked ? 1 : 0,
            this.unlockedAt
          ]
        )
      }

      return { success: true, itemId: this.itemId }
    } catch (error) {
      console.error('❌ Error saving vault item:', error)
      throw error
    }
  }

  /**
   * Convert vault item to JSON representation
   * @param {boolean} includeFullContent - Whether to include full content
   * @returns {Object} - JSON representation
   */
  toJSON (includeFullContent = null) {
    // Auto-determine content inclusion based on unlock status
    const showFullContent =
      includeFullContent !== null ? includeFullContent : this.isUnlocked

    const json = {
      itemId: this.itemId,
      title: this.title,
      description: this.description,
      category: this.category,
      itemType: this.itemType,
      rarity: this.rarity,
      rarityColor: this.getRarityColor(),
      contentPreview: this.contentPreview,
      tags: this.tags,
      estimatedReadTime: this.estimatedReadTime,
      difficulty: this.difficulty,
      relatedConcepts: this.relatedConcepts,
      sources: this.sources,
      isUnlocked: this.isUnlocked,
      unlockedAt: this.unlockedAt,
      isRead: this.isRead,
      readAt: this.readAt,
      isFavorite: this.isFavorite,
      userRating: this.userRating,
      userNotes: this.userNotes,
      accessCount: this.accessCount,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    }

    if (showFullContent) {
      json.contentFull = this.contentFull
    }

    if (!this.isUnlocked) {
      json.unlockConditions = this.unlockConditions
    }

    return json
  }

  // ==========================================
  // STATIC METHODS
  // ==========================================

  /**
   * Find vault item by ID
   * @param {string} itemId - Vault item ID
   * @returns {Promise<VaultItem|null>} - Vault item instance or null
   */
  static async findById (itemId) {
    try {
      const row = await db.get(
        `
                SELECT vi.*, vu.is_unlocked, vu.unlocked_at, vu.is_read, 
                       vu.read_at, vu.is_favorite, vu.user_rating, vu.user_notes
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                WHERE vi.item_id = ?
            `,
        [itemId]
      )

      if (!row) return null

      return VaultItem.fromDatabase(row)
    } catch (error) {
      console.error('❌ Error finding vault item by ID:', error)
      return null
    }
  }

  /**
   * Find vault items by criteria
   * @param {Object} criteria - Search criteria
   * @returns {Promise<Array>} - Array of vault item instances
   */
  static async findByCriteria (criteria = {}) {
    try {
      let whereClause = '1=1'
      let params = []

      if (criteria.category) {
        whereClause += ' AND vi.category = ?'
        params.push(criteria.category)
      }

      if (criteria.itemType) {
        whereClause += ' AND vi.item_type = ?'
        params.push(criteria.itemType)
      }

      if (criteria.rarity) {
        whereClause += ' AND vi.rarity = ?'
        params.push(criteria.rarity)
      }

      if (criteria.isUnlocked !== undefined) {
        whereClause += ' AND vu.is_unlocked = ?'
        params.push(criteria.isUnlocked ? 1 : 0)
      }

      if (criteria.isFavorite !== undefined) {
        whereClause += ' AND vu.is_favorite = ?'
        params.push(criteria.isFavorite ? 1 : 0)
      }

      if (criteria.difficulty) {
        whereClause += ' AND vi.difficulty = ?'
        params.push(criteria.difficulty)
      }

      const orderBy = criteria.orderBy || 'vi.created_at DESC'
      const limit = criteria.limit || 50

      const rows = await db.all(
        `
                SELECT vi.*, vu.is_unlocked, vu.unlocked_at, vu.is_read, 
                       vu.read_at, vu.is_favorite, vu.user_rating, vu.user_notes
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                WHERE ${whereClause}
                ORDER BY ${orderBy}
                LIMIT ?
            `,
        [...params, limit]
      )

      return rows.map(row => VaultItem.fromDatabase(row))
    } catch (error) {
      console.error('❌ Error finding vault items by criteria:', error)
      return []
    }
  }

  /**
   * Create VaultItem instance from database row
   * @param {Object} row - Database row data
   * @returns {VaultItem} - Vault item instance
   */
  static fromDatabase (row) {
    return new VaultItem({
      itemId: row.item_id,
      title: row.title,
      description: row.description,
      category: row.category,
      itemType: row.item_type,
      rarity: row.rarity,
      unlockConditions: JSON.parse(row.unlock_conditions || '{}'),
      contentPreview: row.content_preview,
      contentFull: row.content_full,
      tags: JSON.parse(row.tags || '[]'),
      estimatedReadTime: row.estimated_read_time,
      difficulty: row.difficulty,
      relatedConcepts: JSON.parse(row.related_concepts || '[]'),
      sources: JSON.parse(row.sources || '[]'),
      isUnlocked: !!row.is_unlocked,
      unlockedAt: row.unlocked_at,
      isRead: !!row.is_read,
      readAt: row.read_at,
      isFavorite: !!row.is_favorite,
      userRating: row.user_rating,
      userNotes: row.user_notes || '',
      accessCount: row.access_count || 0,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    })
  }

  /**
   * Get all unlocked vault items
   * @returns {Promise<Array>} - Array of unlocked vault items
   */
  static async getUnlockedItems () {
    return VaultItem.findByCriteria({ isUnlocked: true })
  }

  /**
   * Get vault items by category
   * @param {string} category - Vault category
   * @returns {Promise<Array>} - Array of vault items in category
   */
  static async getByCategory (category) {
    return VaultItem.findByCriteria({ category })
  }

  /**
   * Get favorite vault items
   * @returns {Promise<Array>} - Array of favorited vault items
   */
  static async getFavorites () {
    return VaultItem.findByCriteria({ isFavorite: true, isUnlocked: true })
  }

  /**
   * Check for newly unlockable vault items
   * @returns {Promise<Array>} - Array of newly unlockable items
   */
  static async checkForNewUnlocks () {
    try {
      const lockedItems = await VaultItem.findByCriteria({ isUnlocked: false })
      const newlyUnlockable = []

      for (const item of lockedItems) {
        const unlockStatus = await item.checkUnlockConditions()
        if (unlockStatus.canUnlock) {
          newlyUnlockable.push({
            item,
            trigger: unlockStatus.unlockTrigger
          })
        }
      }

      return newlyUnlockable
    } catch (error) {
      console.error('❌ Error checking for new unlocks:', error)
      return []
    }
  }

  /**
   * Get vault statistics
   * @returns {Promise<Object>} - Vault statistics
   */
  static async getVaultStatistics () {
    try {
      const totalItems = await db.get(
        'SELECT COUNT(*) as count FROM vault_items'
      )
      const unlockedItems = await db.get(
        'SELECT COUNT(*) as count FROM vault_unlocks'
      )
      const readItems = await db.get(
        'SELECT COUNT(*) as count FROM vault_unlocks WHERE is_read = 1'
      )
      const favoriteItems = await db.get(
        'SELECT COUNT(*) as count FROM vault_unlocks WHERE is_favorite = 1'
      )

      const categoryStats = await db.all(`
                SELECT 
                    vi.category,
                    COUNT(vi.item_id) as total_items,
                    COUNT(vu.vault_item_id) as unlocked_items,
                    COUNT(CASE WHEN vu.is_read = 1 THEN 1 END) as read_items
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                GROUP BY vi.category
            `)

      const rarityStats = await db.all(`
                SELECT 
                    vi.rarity,
                    COUNT(vi.item_id) as total_items,
                    COUNT(vu.vault_item_id) as unlocked_items
                FROM vault_items vi
                LEFT JOIN vault_unlocks vu ON vi.item_id = vu.vault_item_id
                GROUP BY vi.rarity
                ORDER BY 
                    CASE vi.rarity 
                        WHEN 'legendary' THEN 5
                        WHEN 'epic' THEN 4
                        WHEN 'rare' THEN 3
                        WHEN 'uncommon' THEN 2
                        WHEN 'common' THEN 1
                    END DESC
            `)

      return {
        total: totalItems.count,
        unlocked: unlockedItems.count,
        read: readItems.count,
        favorites: favoriteItems.count,
        unlockRate:
          totalItems.count > 0 ? unlockedItems.count / totalItems.count : 0,
        readRate:
          unlockedItems.count > 0 ? readItems.count / unlockedItems.count : 0,
        categoryBreakdown: categoryStats,
        rarityBreakdown: rarityStats
      }
    } catch (error) {
      console.error('❌ Error getting vault statistics:', error)
      return {
        total: 0,
        unlocked: 0,
        read: 0,
        favorites: 0,
        unlockRate: 0,
        readRate: 0,
        categoryBreakdown: [],
        rarityBreakdown: []
      }
    }
  }

  /**
   * Initialize vault tables in database
   * @returns {Promise<boolean>} - Success status
   */
  static async initializeTables () {
    try {
      // Main vault items table
      await db.run(`
                CREATE TABLE IF NOT EXISTS vault_items (
                    item_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (category IN ('secret_archives', 'controversy_files', 'beautiful_mind')),
                    item_type TEXT NOT NULL CHECK (item_type IN ('secret', 'tool', 'artifact', 'story', 'wisdom')),
                    rarity TEXT NOT NULL CHECK (rarity IN ('common', 'uncommon', 'rare', 'epic', 'legendary')),
                    unlock_conditions TEXT DEFAULT '{}',
                    content_preview TEXT NOT NULL,
                    content_full TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    estimated_read_time INTEGER DEFAULT 5,
                    difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced', 'expert')),
                    related_concepts TEXT DEFAULT '[]',
                    sources TEXT DEFAULT '[]',
                    is_unlocked BOOLEAN DEFAULT 0,
                    unlocked_at DATETIME,
                    access_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            `)

      // User vault unlock tracking table
      await db.run(`
                CREATE TABLE IF NOT EXISTS vault_unlocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vault_item_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    unlock_trigger TEXT,
                    unlock_phase INTEGER,
                    unlock_week INTEGER,
                    unlock_lesson_id TEXT,
                    is_read BOOLEAN DEFAULT 0,
                    read_at DATETIME,
                    time_spent_reading INTEGER DEFAULT 0,
                    is_favorite BOOLEAN DEFAULT 0,
                    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
                    user_notes TEXT,
                    UNIQUE(vault_item_id),
                    FOREIGN KEY (vault_item_id) REFERENCES vault_items(item_id)
                )
            `)

      // Vault unlock events for analytics
      await db.run(`
                CREATE TABLE IF NOT EXISTS vault_unlock_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    unlock_conditions_met TEXT,
                    unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (item_id) REFERENCES vault_items(item_id)
                )
            `)

      console.log('✅ Vault tables initialized')
      return true
    } catch (error) {
      console.error('❌ Error initializing vault tables:', error)
      return false
    }
  }
}

module.exports = VaultItem
