/**
 * Neural Odyssey Learning Controller
 *
 * Handles all learning-related operations including:
 * - Progress tracking through the ML path phases/weeks/lessons
 * - Daily session management with rotation (math, coding, visual, applications)
 * - Quest/project completions with code storage
 * - Skill points and achievements
 * - Spaced repetition system
 * - Learning analytics and insights
 *
 * Connects to: backend/routes/learning.js + backend/config/db.js
 * Uses: SQLite tables from scripts/init-db.js
 *
 * Author: Neural Explorer
 */

const db = require('../config/db')
const { validationResult } = require('express-validator')
const moment = require('moment')
const _ = require('lodash')

class LearningController {
  // ==========================================
  // PROGRESS TRACKING
  // ==========================================

  /**
   * Get overall learning progress summary
   * GET /api/v1/learning/progress
   */
  async getProgress (req, res, next) {
    try {
      const { phase, week } = req.query

      let whereClause = '1=1'
      let params = []

      if (phase) {
        whereClause += ' AND phase = ?'
        params.push(phase)
      }

      if (week) {
        whereClause += ' AND week = ?'
        params.push(week)
      }

      // Get learning progress
      const progress = await db.query(
        `
                SELECT 
                    phase,
                    week,
                    lesson_id,
                    lesson_title,
                    lesson_type,
                    status,
                    completion_percentage,
                    time_spent_minutes,
                    completed_at,
                    mastery_score,
                    notes,
                    created_at,
                    updated_at
                FROM learning_progress 
                WHERE ${whereClause}
                ORDER BY phase, week, lesson_id
            `,
        params
      )

      // Get current user profile
      const profile = await db.get(`
                SELECT 
                    current_phase,
                    current_week,
                    total_study_minutes,
                    current_streak_days,
                    longest_streak_days,
                    last_activity_date
                FROM user_profile WHERE id = 1
            `)

      // Calculate statistics
      const stats = {
        total_lessons: progress.length,
        completed_lessons: progress.filter(p =>
          ['completed', 'mastered'].includes(p.status)
        ).length,
        mastered_lessons: progress.filter(p => p.status === 'mastered').length,
        in_progress_lessons: progress.filter(p => p.status === 'in_progress')
          .length,
        total_time_spent: progress.reduce(
          (sum, p) => sum + (p.time_spent_minutes || 0),
          0
        ),
        average_mastery_score: progress
          .filter(p => p.mastery_score)
          .reduce((sum, p, i, arr) => sum + p.mastery_score / arr.length, 0),
        completion_rate:
          progress.length > 0
            ? (
                (progress.filter(p =>
                  ['completed', 'mastered'].includes(p.status)
                ).length /
                  progress.length) *
                100
              ).toFixed(1)
            : 0
      }

      res.json({
        success: true,
        data: {
          profile,
          progress,
          statistics: stats
        }
      })
    } catch (error) {
      next(error)
    }
  }

  /**
   * Update lesson progress
   * PUT /api/v1/learning/progress/:lessonId
   */
  async updateProgress (req, res, next) {
    try {
      const errors = validationResult(req)
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          message: 'Validation failed',
          errors: errors.array()
        })
      }

      const { lessonId } = req.params
      const {
        status,
        completion_percentage,
        time_spent_minutes,
        mastery_score,
        notes
      } = req.body

      // Get current lesson data
      const currentLesson = await db.get(
        `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
        [lessonId]
      )

      if (!currentLesson) {
        return res.status(404).json({
          success: false,
          message: 'Lesson not found'
        })
      }

      // Build update object
      const updateData = {
        status: status || currentLesson.status,
        completion_percentage:
          completion_percentage !== undefined
            ? completion_percentage
            : currentLesson.completion_percentage,
        time_spent_minutes:
          (currentLesson.time_spent_minutes || 0) + (time_spent_minutes || 0),
        mastery_score:
          mastery_score !== undefined
            ? mastery_score
            : currentLesson.mastery_score,
        notes: notes !== undefined ? notes : currentLesson.notes,
        updated_at: new Date().toISOString()
      }

      // Set completion date if newly completed/mastered
      if (
        ['completed', 'mastered'].includes(status) &&
        !currentLesson.completed_at
      ) {
        updateData.completed_at = new Date().toISOString()
      }

      const result = await db.run(
        `
            UPDATE learning_progress 
            SET status = ?, completion_percentage = ?, time_spent_minutes = ?, 
                mastery_score = ?, notes = ?, completed_at = ?, updated_at = ?
            WHERE lesson_id = ?
        `,
        [
          updateData.status,
          updateData.completion_percentage,
          updateData.time_spent_minutes,
          updateData.mastery_score,
          updateData.notes,
          updateData.completed_at || currentLesson.completed_at,
          updateData.updated_at,
          lessonId
        ]
      )

      // Award skill points for significant progress
      if (
        ['completed', 'mastered'].includes(status) &&
        currentLesson.status !== status
      ) {
        await this.awardSkillPoints(currentLesson.lesson_type, status, lessonId)
      }

      // Update user profile
      await this.updateUserProfile(time_spent_minutes)

      // Check for vault unlocks
      const vaultUnlocks = await this.checkVaultUnlocks(
        currentLesson.phase,
        currentLesson.week,
        lessonId
      )

      // Get updated lesson data
      const updatedLesson = await db.get(
        `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
        [lessonId]
      )

      res.json({
        success: true,
        message: 'Progress updated successfully',
        data: {
          lesson: updatedLesson,
          vault_unlocks: vaultUnlocks || []
        }
      })
    } catch (error) {
      next(error)
    }
  }

  // ==========================================
  // DAILY SESSIONS
  // ==========================================

  /**
   * Start a new learning session
   * POST /api/v1/learning/sessions/start
   */
  async startSession (req, res, next) {
    try {
      const { session_type, energy_level, goals } = req.body

      const validTypes = [
        'math',
        'coding',
        'visual_projects',
        'real_applications'
      ]
      if (!validTypes.includes(session_type)) {
        return res.status(400).json({
          success: false,
          message: 'Invalid session type',
          validTypes
        })
      }

      const result = await db.run(
        `
            INSERT INTO daily_sessions (session_date, session_type, start_time, energy_level, planned_duration_minutes)
            VALUES (?, ?, ?, ?, ?)
        `,
        [
          moment().format('YYYY-MM-DD'),
          session_type,
          new Date().toISOString(),
          energy_level || 8,
          25 // Default Pomodoro length
        ]
      )

      const session = await db.get(
        `
            SELECT * FROM daily_sessions WHERE id = ?
        `,
        [result.lastID]
      )

      res.json({
        success: true,
        message: 'Session started successfully',
        data: session
      })
    } catch (error) {
      next(error)
    }
  }

  /**
   * End a learning session
   * PUT /api/v1/learning/sessions/:sessionId/end
   */
  async endSession (req, res, next) {
    try {
      const { sessionId } = req.params
      const { focus_score, session_notes, completed_goals } = req.body

      const session = await db.get(
        `
            SELECT * FROM daily_sessions WHERE id = ?
        `,
        [sessionId]
      )

      if (!session) {
        return res.status(404).json({
          success: false,
          message: 'Session not found'
        })
      }

      const startTime = new Date(session.start_time)
      const endTime = new Date()
      const durationMinutes = Math.round((endTime - startTime) / (1000 * 60))

      await db.run(
        `
            UPDATE daily_sessions 
            SET end_time = ?, actual_duration_minutes = ?, focus_score = ?, 
                session_notes = ?, completed_goals = ?
            WHERE id = ?
        `,
        [
          endTime.toISOString(),
          durationMinutes,
          focus_score || null,
          session_notes || '',
          JSON.stringify(completed_goals || []),
          sessionId
        ]
      )

      // Update user profile with study time
      await this.updateUserProfile(durationMinutes)

      const updatedSession = await db.get(
        `
            SELECT * FROM daily_sessions WHERE id = ?
        `,
        [sessionId]
      )

      res.json({
        success: true,
        message: 'Session ended successfully',
        data: updatedSession
      })
    } catch (error) {
      next(error)
    }
  }

  /**
   * Get today's sessions and recommendations
   * GET /api/v1/learning/sessions/today
   */
  async getTodaySessions (req, res, next) {
    try {
      const today = moment().format('YYYY-MM-DD')

      const sessions = await db.query(
        `
                SELECT * FROM daily_sessions 
                WHERE session_date = ?
                ORDER BY start_time DESC
            `,
        [today]
      )

      // Calculate today's progress
      const totalTime = sessions.reduce(
        (sum, s) => sum + (s.duration_minutes || 0),
        0
      )
      const avgFocus =
        sessions.length > 0
          ? sessions.reduce((sum, s) => sum + (s.focus_score || 0), 0) /
            sessions.length
          : 0

      // Recommend next session type based on daily rotation
      const sessionTypes = [
        'math',
        'coding',
        'visual_projects',
        'real_applications'
      ]
      const usedTypes = sessions.map(s => s.session_type)
      const nextRecommended =
        sessionTypes.find(type => !usedTypes.includes(type)) || sessionTypes[0]

      res.json({
        success: true,
        data: {
          date: today,
          sessions,
          summary: {
            total_time_minutes: totalTime,
            session_count: sessions.length,
            average_focus: avgFocus,
            recommended_next_type: nextRecommended
          }
        }
      })
    } catch (error) {
      next(error)
    }
  }

  // ==========================================
  // QUEST/PROJECT MANAGEMENT
  // ==========================================

  /**
   * Submit a quest completion
   * POST /api/v1/learning/quests/submit
   */
  async submitQuest (req, res, next) {
    try {
      const errors = validationResult(req)
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          message: 'Validation failed',
          errors: errors.array()
        })
      }

      const {
        quest_id,
        quest_title,
        quest_type,
        phase,
        week,
        difficulty_level,
        code_solution,
        execution_result,
        time_to_complete_minutes,
        self_reflection,
        status = 'completed'
      } = req.body

      // Check if quest already exists
      const existingQuest = await db.get(
        `
                SELECT * FROM quest_completions WHERE quest_id = ?
            `,
        [quest_id]
      )

      if (existingQuest) {
        // Update existing quest
        const result = await db.run(
          `
                    UPDATE quest_completions 
                    SET status = ?, code_solution = ?, execution_result = ?, 
                        time_to_complete_minutes = ?, attempts_count = attempts_count + 1,
                        self_reflection = ?, completed_at = ?
                    WHERE quest_id = ?
                `,
          [
            status,
            code_solution,
            execution_result,
            time_to_complete_minutes,
            self_reflection,
            new Date().toISOString(),
            quest_id
          ]
        )
      } else {
        // Create new quest completion
        const result = await db.run(
          `
                    INSERT INTO quest_completions (
                        quest_id, quest_title, quest_type, phase, week, difficulty_level,
                        status, code_solution, execution_result, time_to_complete_minutes,
                        self_reflection, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                `,
          [
            quest_id,
            quest_title,
            quest_type,
            phase,
            week,
            difficulty_level,
            status,
            code_solution,
            execution_result,
            time_to_complete_minutes,
            self_reflection,
            new Date().toISOString()
          ]
        )
      }

      // Award skill points based on quest type and difficulty
      await this.awardQuestSkillPoints(
        quest_type,
        difficulty_level,
        status,
        quest_id
      )

      // Generate AI mentor feedback (placeholder for now)
      const mentorFeedback = await this.generateMentorFeedback(
        code_solution,
        execution_result,
        quest_type
      )

      // Update mentor feedback
      await db.run(
        `
                UPDATE quest_completions SET mentor_feedback = ? WHERE quest_id = ?
            `,
        [mentorFeedback, quest_id]
      )

      res.json({
        success: true,
        message: 'Quest submitted successfully',
        data: {
          quest_id,
          status,
          mentor_feedback: mentorFeedback
        }
      })
    } catch (error) {
      next(error)
    }
  }

  /**
   * Get user's quest history
   * GET /api/v1/learning/quests
   */
  async getQuests (req, res, next) {
    try {
      const { phase, status, limit = 50 } = req.query

      let whereClause = '1=1'
      let params = []

      if (phase) {
        whereClause += ' AND phase = ?'
        params.push(phase)
      }

      if (status) {
        whereClause += ' AND status = ?'
        params.push(status)
      }

      params.push(limit)

      const quests = await db.query(
        `
                SELECT * FROM quest_completions 
                WHERE ${whereClause}
                ORDER BY completed_at DESC
                LIMIT ?
            `,
        params
      )

      res.json({
        success: true,
        data: quests
      })
    } catch (error) {
      next(error)
    }
  }

  // ==========================================
  // SPACED REPETITION
  // ==========================================

  /**
   * Get items due for review
   * GET /api/v1/learning/review
   */
  async getReviewItems (req, res, next) {
    try {
      const today = moment().format('YYYY-MM-DD')

      const reviewItems = await db.query(
        `
                SELECT * FROM spaced_repetition 
                WHERE next_review_date <= ?
                ORDER BY next_review_date ASC, repetition_count ASC
                LIMIT 10
            `,
        [today]
      )

      res.json({
        success: true,
        data: {
          review_items: reviewItems,
          count: reviewItems.length,
          date: today
        }
      })
    } catch (error) {
      next(error)
    }
  }

  /**
   * Submit spaced repetition review
   * POST /api/v1/learning/review/:conceptId
   */
  async submitReview (req, res, next) {
    try {
      const { conceptId } = req.params
      const { quality_score } = req.body // 0-5 scale

      if (quality_score < 0 || quality_score > 5) {
        return res.status(400).json({
          success: false,
          message: 'Quality score must be between 0 and 5'
        })
      }

      const item = await db.get(
        `
            SELECT * FROM spaced_repetition WHERE concept_id = ?
        `,
        [conceptId]
      )

      if (!item) {
        return res.status(404).json({
          success: false,
          message: 'Review item not found'
        })
      }

      // SM-2 algorithm for spaced repetition
      const { newInterval, newDifficulty } = this.calculateSM2(
        item.interval_days,
        item.easiness_factor,
        quality_score
      )

      const nextReviewDate = moment()
        .add(newInterval, 'days')
        .format('YYYY-MM-DD')

      await db.run(
        `
            UPDATE spaced_repetition 
            SET interval_days = ?, easiness_factor = ?, repetitions = repetitions + 1,
                next_review_date = ?, last_review_date = ?, quality_score = ?
            WHERE concept_id = ?
        `,
        [
          newInterval,
          newDifficulty,
          nextReviewDate,
          moment().format('YYYY-MM-DD'),
          quality_score,
          conceptId
        ]
      )

      res.json({
        success: true,
        message: 'Review submitted successfully',
        data: {
          concept_id: conceptId,
          next_review_date: nextReviewDate,
          interval_days: newInterval
        }
      })
    } catch (error) {
      next(error)
    }
  }

  // ==========================================
  // ANALYTICS
  // ==========================================

  /**
   * Get learning analytics
   * GET /api/v1/learning/analytics
   */
  async getAnalytics (req, res, next) {
    try {
      const { period = '30' } = req.query // days
      const startDate = moment().subtract(period, 'days').format('YYYY-MM-DD')

      // Session analytics
      const sessionStats = await db.get(
        `
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(duration_minutes) as total_time,
                    AVG(duration_minutes) as avg_session_length,
                    AVG(focus_score) as avg_focus_score,
                    AVG(energy_level) as avg_energy_level
                FROM daily_sessions 
                WHERE session_date >= ?
            `,
        [startDate]
      )

      // Progress analytics
      const progressStats = await db.get(
        `
                SELECT 
                    COUNT(CASE WHEN status = 'completed' OR status = 'mastered' THEN 1 END) as completed_lessons,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                    AVG(time_spent_minutes) as avg_time_per_lesson,
                    AVG(mastery_score) as avg_mastery_score
                FROM learning_progress 
                WHERE updated_at >= ?
            `,
        [startDate]
      )

      // Skill distribution
      const skillDistribution = await db.query(
        `
                SELECT 
                    category,
                    SUM(points_earned) as total_points,
                    COUNT(*) as achievements
                FROM skill_points 
                WHERE earned_at >= ?
                GROUP BY category
                ORDER BY total_points DESC
            `,
        [startDate]
      )

      // Daily activity heatmap
      const dailyActivity = await db.query(
        `
                SELECT 
                    session_date,
                    COUNT(*) as session_count,
                    SUM(duration_minutes) as total_time,
                    AVG(focus_score) as avg_focus
                FROM daily_sessions 
                WHERE session_date >= ?
                GROUP BY session_date
                ORDER BY session_date
            `,
        [startDate]
      )

      res.json({
        success: true,
        data: {
          period_days: period,
          sessions: sessionStats,
          progress: progressStats,
          skills: skillDistribution,
          daily_activity: dailyActivity,
          generated_at: new Date().toISOString()
        }
      })
    } catch (error) {
      next(error)
    }
  }

  // ==========================================
  // HELPER METHODS
  // ==========================================

  async awardSkillPoints (lessonType, status, lessonId) {
    const pointsMap = {
      theory: { completed: 10, mastered: 25 },
      math: { completed: 15, mastered: 30 },
      visual: { completed: 12, mastered: 28 },
      coding: { completed: 20, mastered: 40 }
    }

    const points = pointsMap[lessonType]?.[status] || 10
    const category =
      lessonType === 'coding'
        ? 'programming'
        : lessonType === 'math'
        ? 'mathematics'
        : 'theory'

    await db.run(
      `
        INSERT INTO skill_points (category, points_earned, reason, related_lesson_id)
        VALUES (?, ?, ?, ?)
    `,
      [category, points, `Lesson ${status}: ${lessonId}`, lessonId]
    )
  }

  async getRecommendedQuests (phase, week) {
    try {
      // Get user's completed quests to avoid duplicates
      const completedQuests = await db.query(
        `
            SELECT quest_id FROM quest_completions WHERE phase = ? AND week = ?
        `,
        [phase, week]
      )

      const completedIds = new Set(completedQuests.map(q => q.quest_id))

      // Return quest recommendations based on current progress
      // This would integrate with Quest model - placeholder for now
      return []
    } catch (error) {
      console.error('Error getting recommended quests:', error)
      return []
    }
  }

  async awardQuestSkillPoints (questType, difficulty, status, questId) {
    const basePoints = {
      coding_exercise: 15,
      implementation_project: 25,
      theory_quiz: 10,
      practical_application: 20
    }

    const points =
      (basePoints[questType] || 10) *
      difficulty *
      (status === 'mastered' ? 1.5 : 1)

    await db.run(
      `
            INSERT INTO skill_points (category, points_earned, reason, related_quest_id)
            VALUES (?, ?, ?, ?)
        `,
      ['programming', Math.round(points), `Quest ${status}`, questId]
    )
  }

  async updateUserProfile (additionalMinutes = 0) {
    await db.run(
      `
            UPDATE user_profile 
            SET total_study_minutes = total_study_minutes + ?,
                last_activity_date = date('now')
            WHERE id = 1
        `,
      [additionalMinutes]
    )
  }

  async checkVaultUnlocks (phase, week, lessonId) {
    try {
      // Import vault controller to check unlocks
      const VaultController = require('./vaultController')

      // Check for any vault items that should be unlocked
      const newUnlocks = []

      // Get all vault items that aren't already unlocked
      const existingUnlocks = await db.query(`
            SELECT vault_item_id FROM vault_unlocks
        `)
      const unlockedIds = new Set(existingUnlocks.map(u => u.vault_item_id))

      // Check lesson completion unlock conditions
      const lessonUnlocks = await this.checkLessonUnlockConditions(
        phase,
        week,
        lessonId,
        unlockedIds
      )
      newUnlocks.push(...lessonUnlocks)

      // Process any new unlocks
      for (const unlock of newUnlocks) {
        await db.run(
          `
                INSERT INTO vault_unlocks (vault_item_id, category, unlock_trigger, unlock_phase, unlock_week, unlock_lesson_id)
                VALUES (?, ?, ?, ?, ?, ?)
            `,
          [unlock.id, unlock.category, 'lesson_complete', phase, week, lessonId]
        )
      }

      return newUnlocks
    } catch (error) {
      console.error('Error checking vault unlocks:', error)
      return []
    }
  }

  async checkLessonUnlockConditions (phase, week, lessonId, unlockedIds) {
    // This would check against vault-items.json unlock conditions
    // For now, return empty array - will be implemented when vault system is connected
    return []
  }

  async generateMentorFeedback (code, result, questType) {
    // Placeholder for AI mentor feedback
    // In a real implementation, this could use OpenAI API or local ML model
    const feedbackTemplates = {
      coding_exercise:
        'Great work on the implementation! Consider optimizing for better performance.',
      implementation_project:
        'Excellent project completion! Your code structure shows good understanding.',
      theory_quiz: 'Well done! Your theoretical understanding is solid.',
      practical_application:
        'Impressive practical application! Consider exploring edge cases.'
    }

    return (
      feedbackTemplates[questType] ||
      'Good effort! Keep practicing to improve your skills.'
    )
  }

  calculateSM2 (interval, difficulty, quality) {
    // SM-2 spaced repetition algorithm
    let newDifficulty = difficulty
    let newInterval = interval

    if (quality >= 3) {
      newInterval = interval === 1 ? 6 : Math.round(interval * newDifficulty)
      newDifficulty =
        newDifficulty + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    } else {
      newInterval = 1
      newDifficulty = Math.max(1.3, newDifficulty - 0.2)
    }

    return {
      newInterval: Math.max(1, newInterval),
      newDifficulty: Math.max(1.3, Math.min(2.5, newDifficulty))
    }
  }
}

module.exports = new LearningController()
