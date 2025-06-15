/**
 * Neural Odyssey Learning Routes
 *
 * RESTful API endpoints for managing learning progress, quests, sessions,
 * and all learning-related data in the Neural Odyssey platform.
 *
 * Endpoints:
 * - GET /progress - Get overall learning progress
 * - GET /progress/phase/:phase - Get phase-specific progress
 * - GET /lessons - Get all lessons with filters
 * - GET /lessons/:id - Get specific lesson details
 * - PUT /lessons/:id/progress - Update lesson progress
 * - POST /lessons/:id/complete - Mark lesson as completed
 * - GET /quests - Get quest/project completions
 * - POST /quests - Create new quest completion
 * - PUT /quests/:id - Update quest completion
 * - GET /sessions - Get daily learning sessions
 * - POST /sessions - Start new learning session
 * - PUT /sessions/:id - Update/end learning session
 * - GET /spaced-repetition - Get items due for review
 * - POST /spaced-repetition/:id/review - Record spaced repetition review
 * - GET /knowledge-graph - Get knowledge connections
 * - POST /knowledge-graph - Add knowledge connection
 * - GET /next-lesson - Get next lesson recommendation
 * - GET /analytics - Get learning analytics
 * - GET /streak - Get current learning streak
 * - POST /reset-progress - Reset learning progress (with confirmation)
 *
 * Author: Neural Explorer
 */

const express = require('express')
const { body, param, query, validationResult } = require('express-validator')
const router = express.Router()
const db = require('../config/db')
const moment = require('moment')
const fs = require('fs')
const path = require('path')
const {
  asyncErrorHandler,
  ProgressNotFoundError,
  QuestNotFoundError
} = require('../middleware/errorHandler')

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

// ==========================================
// PROGRESS TRACKING ENDPOINTS
// ==========================================

// GET /api/v1/learning/progress - Get overall learning progress
router.get(
  '/progress',
  asyncErrorHandler(async (req, res) => {
    try {
      const { phase, week } = req.query

      // Get user profile
      const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

      if (!profile) {
        // Create default profile if none exists
        await db.run(`
                INSERT INTO user_profile (
                    id, username, created_at, updated_at
                ) VALUES (1, 'Neural Explorer', datetime('now'), datetime('now'))
            `)

        const newProfile = await db.get(
          'SELECT * FROM user_profile WHERE id = 1'
        )
        profile = newProfile
      }

      // Build where clause for filtering
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
      const progressData = await db.all(
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
                mastery_score,
                completed_at,
                notes,
                difficulty_rating,
                created_at,
                updated_at
            FROM learning_progress 
            WHERE ${whereClause}
            ORDER BY phase, week, lesson_id
        `,
        params
      )

      // Calculate summary statistics
      const summary = {
        totalLessons: progressData.length,
        completedLessons: progressData.filter(
          p => p.status === 'completed' || p.status === 'mastered'
        ).length,
        masteredLessons: progressData.filter(p => p.status === 'mastered')
          .length,
        inProgressLessons: progressData.filter(p => p.status === 'in_progress')
          .length,
        totalStudyTime: progressData.reduce(
          (sum, p) => sum + (p.time_spent_minutes || 0),
          0
        ),
        averageMasteryScore:
          progressData.filter(p => p.mastery_score).length > 0
            ? progressData
                .filter(p => p.mastery_score)
                .reduce((sum, p) => sum + p.mastery_score, 0) /
              progressData.filter(p => p.mastery_score).length
            : null,
        currentPhase: profile.current_phase,
        currentWeek: profile.current_week,
        streakDays: profile.current_streak_days
      }

      // Group progress by phase and week
      const progressByPhase = {}
      progressData.forEach(lesson => {
        if (!progressByPhase[lesson.phase]) {
          progressByPhase[lesson.phase] = {}
        }
        if (!progressByPhase[lesson.phase][lesson.week]) {
          progressByPhase[lesson.phase][lesson.week] = []
        }
        progressByPhase[lesson.phase][lesson.week].push(lesson)
      })

      res.json({
        success: true,
        data: {
          profile,
          summary,
          progressByPhase,
          lessons: progressData
        }
      })
    } catch (error) {
      throw error
    }
  })
)

// GET /api/v1/learning/progress/phase/:phase - Get phase-specific progress
router.get(
  '/progress/phase/:phase',
  [
    param('phase')
      .isInt({ min: 1, max: 4 })
      .withMessage('Phase must be between 1 and 4')
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase } = req.params

    const phaseProgress = await db.all(
      `
            SELECT * FROM learning_progress 
            WHERE phase = ?
            ORDER BY week, lesson_id
        `,
      [phase]
    )

    const phaseStats = {
      phase: parseInt(phase),
      totalLessons: phaseProgress.length,
      completedLessons: phaseProgress.filter(
        p => p.status === 'completed' || p.status === 'mastered'
      ).length,
      totalTime: phaseProgress.reduce(
        (sum, p) => sum + (p.time_spent_minutes || 0),
        0
      ),
      averageScore:
        phaseProgress.filter(p => p.mastery_score).length > 0
          ? phaseProgress
              .filter(p => p.mastery_score)
              .reduce((sum, p) => sum + p.mastery_score, 0) /
            phaseProgress.filter(p => p.mastery_score).length
          : null
    }

    res.json({
      success: true,
      data: {
        stats: phaseStats,
        lessons: phaseProgress
      }
    })
  })
)

// ==========================================
// LESSON MANAGEMENT ENDPOINTS
// ==========================================

// GET /api/v1/learning/lessons - Get all lessons with filters
router.get(
  '/lessons',
  [
    query('phase').optional().isInt({ min: 1, max: 4 }),
    query('week').optional().isInt({ min: 1 }),
    query('status')
      .optional()
      .isIn(['not_started', 'in_progress', 'completed', 'mastered']),
    query('type')
      .optional()
      .isIn(['theory', 'math', 'coding', 'visual', 'project']),
    query('limit')
      .optional()
      .isInt({ min: 1, max: 100 })
      .withMessage('Limit must be between 1 and 100'),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase, week, status, type, limit = 50, offset = 0 } = req.query

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

    if (status) {
      whereClause += ' AND status = ?'
      params.push(status)
    }

    if (type) {
      whereClause += ' AND lesson_type = ?'
      params.push(type)
    }

    const lessons = await db.all(
      `
            SELECT * FROM learning_progress 
            WHERE ${whereClause}
            ORDER BY phase, week, lesson_id
            LIMIT ? OFFSET ?
        `,
      [...params, limit, offset]
    )

    const total = await db.get(
      `
            SELECT COUNT(*) as count FROM learning_progress 
            WHERE ${whereClause}
        `,
      params
    )

    res.json({
      success: true,
      data: {
        lessons,
        pagination: {
          total: total.count,
          limit: parseInt(limit),
          offset: parseInt(offset),
          pages: Math.ceil(total.count / limit)
        }
      }
    })
  })
)

// GET /api/v1/learning/lessons/:id - Get specific lesson details
router.get(
  '/lessons/:id',
  [param('id').notEmpty().withMessage('Lesson ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    const lesson = await db.get(
      `
            SELECT * FROM learning_progress 
            WHERE lesson_id = ?
        `,
      [id]
    )

    if (!lesson) {
      throw new ProgressNotFoundError(id)
    }

    // Get related quest completions
    const relatedQuests = await db.all(
      `
            SELECT * FROM quest_completions 
            WHERE phase = ? AND week = ?
            ORDER BY completed_at DESC
        `,
      [lesson.phase, lesson.week]
    )

    // Get spaced repetition data if exists
    const spacedRepetition = await db.get(
      `
            SELECT * FROM spaced_repetition 
            WHERE concept_id = ?
        `,
      [id]
    )

    res.json({
      success: true,
      data: {
        lesson,
        relatedQuests,
        spacedRepetition
      }
    })
  })
)

// PUT /api/v1/learning/lessons/:id/progress - Update lesson progress
router.put(
  '/lessons/:id/progress',
  [
    param('id').notEmpty().withMessage('Lesson ID is required'),
    body('status')
      .optional()
      .isIn(['not_started', 'in_progress', 'completed', 'mastered']),
    body('completion_percentage').optional().isInt({ min: 0, max: 100 }),
    body('time_spent_minutes').optional().isInt({ min: 0 }),
    body('mastery_score').optional().isFloat({ min: 0, max: 1 }),
    body('notes').optional().isString().isLength({ max: 2000 }),
    body('difficulty_rating').optional().isInt({ min: 1, max: 5 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const {
      status,
      completion_percentage,
      time_spent_minutes,
      mastery_score,
      notes,
      difficulty_rating
    } = req.body

    // Check if lesson exists
    const existingLesson = await db.get(
      `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
      [id]
    )

    if (!existingLesson) {
      throw new ProgressNotFoundError(id)
    }

    // Build update query dynamically
    const updates = []
    const values = []

    if (status !== undefined) {
      updates.push('status = ?')
      values.push(status)

      // Set completion date if status is completed or mastered
      if (status === 'completed' || status === 'mastered') {
        updates.push('completed_at = datetime("now")')
      }
    }

    if (completion_percentage !== undefined) {
      updates.push('completion_percentage = ?')
      values.push(completion_percentage)
    }

    if (time_spent_minutes !== undefined) {
      updates.push('time_spent_minutes = time_spent_minutes + ?')
      values.push(time_spent_minutes)
    }

    if (mastery_score !== undefined) {
      updates.push('mastery_score = ?')
      values.push(mastery_score)
    }

    if (notes !== undefined) {
      updates.push('notes = ?')
      values.push(notes)
    }

    if (difficulty_rating !== undefined) {
      updates.push('difficulty_rating = ?')
      values.push(difficulty_rating)
    }

    if (updates.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No valid fields to update'
      })
    }

    updates.push('updated_at = datetime("now")')
    values.push(id)

    const sql = `UPDATE learning_progress SET ${updates.join(
      ', '
    )} WHERE lesson_id = ?`

    await db.run(sql, values)

    // Update user profile study time if time was added
    if (time_spent_minutes > 0) {
      await db.run(
        `
                UPDATE user_profile 
                SET total_study_minutes = total_study_minutes + ?,
                    last_activity_date = date('now'),
                    updated_at = datetime('now')
                WHERE id = 1
            `,
        [time_spent_minutes]
      )
    }

    // Get updated lesson
    const updatedLesson = await db.get(
      `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
      [id]
    )

    res.json({
      success: true,
      message: 'Lesson progress updated successfully',
      data: updatedLesson
    })
  })
)

// POST /api/v1/learning/lessons/:id/complete - Mark lesson as completed
router.post(
  '/lessons/:id/complete',
  [
    param('id').notEmpty().withMessage('Lesson ID is required'),
    body('time_spent_minutes')
      .isInt({ min: 1 })
      .withMessage('Time spent is required'),
    body('mastery_score').optional().isFloat({ min: 0, max: 1 }),
    body('notes').optional().isString().isLength({ max: 2000 }),
    body('difficulty_rating').optional().isInt({ min: 1, max: 5 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const { time_spent_minutes, mastery_score, notes, difficulty_rating } =
      req.body

    // Check if lesson exists
    const lesson = await db.get(
      `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
      [id]
    )

    if (!lesson) {
      throw new ProgressNotFoundError(id)
    }

    // Mark as completed
    await db.run(
      `
            UPDATE learning_progress 
            SET status = 'completed',
                completion_percentage = 100,
                time_spent_minutes = time_spent_minutes + ?,
                mastery_score = ?,
                notes = ?,
                difficulty_rating = ?,
                completed_at = datetime('now'),
                updated_at = datetime('now')
            WHERE lesson_id = ?
        `,
      [
        time_spent_minutes,
        mastery_score || null,
        notes || lesson.notes,
        difficulty_rating || null,
        id
      ]
    )

    // Update user profile
    await db.run(
      `
            UPDATE user_profile 
            SET total_study_minutes = total_study_minutes + ?,
                last_activity_date = date('now'),
                updated_at = datetime('now')
            WHERE id = 1
        `,
      [time_spent_minutes]
    )

    // Update daily session
    const today = moment().format('YYYY-MM-DD')
    await db.run(
      `
            INSERT OR REPLACE INTO daily_sessions (
                session_date, session_type, duration_minutes, lessons_completed, focus_score
            ) VALUES (
                ?, 
                ?, 
                COALESCE((SELECT duration_minutes FROM daily_sessions WHERE session_date = ?), 0) + ?,
                COALESCE((SELECT lessons_completed FROM daily_sessions WHERE session_date = ?), 0) + 1,
                ?
            )
        `,
      [
        today,
        lesson.lesson_type,
        today,
        time_spent_minutes,
        today,
        difficulty_rating || 7
      ]
    )

    // Add to spaced repetition system
    await db.run(
      `
            INSERT OR REPLACE INTO spaced_repetition (
                concept_id, concept_title, next_review_date, created_at
            ) VALUES (?, ?, date('now', '+1 day'), datetime('now'))
        `,
      [id, lesson.lesson_title]
    )

    const updatedLesson = await db.get(
      `
            SELECT * FROM learning_progress WHERE lesson_id = ?
        `,
      [id]
    )

    res.json({
      success: true,
      message: 'Lesson completed successfully!',
      data: updatedLesson
    })
  })
)

// ==========================================
// QUEST MANAGEMENT ENDPOINTS
// ==========================================

// GET /api/v1/learning/quests - Get quest/project completions
router.get(
  '/quests',
  [
    query('phase').optional().isInt({ min: 1, max: 4 }),
    query('week').optional().isInt({ min: 1 }),
    query('status').optional().isIn(['attempted', 'completed', 'mastered']),
    query('type')
      .optional()
      .isIn([
        'coding_exercise',
        'implementation_project',
        'theory_quiz',
        'practical_application'
      ]),
    query('limit').optional().isInt({ min: 1, max: 50 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase, week, status, type, limit = 20, offset = 0 } = req.query

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

    if (status) {
      whereClause += ' AND status = ?'
      params.push(status)
    }

    if (type) {
      whereClause += ' AND quest_type = ?'
      params.push(type)
    }

    const quests = await db.all(
      `
            SELECT * FROM quest_completions 
            WHERE ${whereClause}
            ORDER BY completed_at DESC
            LIMIT ? OFFSET ?
        `,
      [...params, limit, offset]
    )

    const total = await db.get(
      `
            SELECT COUNT(*) as count FROM quest_completions 
            WHERE ${whereClause}
        `,
      params
    )

    res.json({
      success: true,
      data: {
        quests,
        pagination: {
          total: total.count,
          limit: parseInt(limit),
          offset: parseInt(offset)
        }
      }
    })
  })
)

// POST /api/v1/learning/quests - Create new quest completion
router.post(
  '/quests',
  [
    body('quest_id').notEmpty().withMessage('Quest ID is required'),
    body('quest_title').notEmpty().withMessage('Quest title is required'),
    body('quest_type').isIn([
      'coding_exercise',
      'implementation_project',
      'theory_quiz',
      'practical_application'
    ]),
    body('phase').isInt({ min: 1, max: 4 }),
    body('week').isInt({ min: 1 }),
    body('status').isIn(['attempted', 'completed', 'mastered']),
    body('time_to_complete_minutes').isInt({ min: 1 }),
    body('user_code').optional().isString(),
    body('final_solution').optional().isString(),
    body('test_results').optional().isString(),
    body('scores').optional().isObject(),
    body('learning_notes').optional().isString().isLength({ max: 2000 }),
    body('challenges_faced').optional().isString().isLength({ max: 1000 }),
    body('insights_gained').optional().isString().isLength({ max: 1000 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      quest_id,
      quest_title,
      quest_type,
      phase,
      week,
      status,
      time_to_complete_minutes,
      user_code,
      final_solution,
      test_results,
      scores = {},
      learning_notes,
      challenges_faced,
      insights_gained
    } = req.body

    // Insert quest completion
    const result = await db.run(
      `
            INSERT INTO quest_completions (
                quest_id, quest_title, quest_type, phase, week,
                status, time_to_complete_minutes,
                user_code, final_solution, test_results,
                code_quality_score, creativity_score, efficiency_score,
                learning_notes, challenges_faced, insights_gained,
                completed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                     datetime('now'), datetime('now'))
        `,
      [
        quest_id,
        quest_title,
        quest_type,
        phase,
        week,
        status,
        time_to_complete_minutes,
        user_code || null,
        final_solution || null,
        test_results || null,
        scores.quality || null,
        scores.creativity || null,
        scores.efficiency || null,
        learning_notes || null,
        challenges_faced || null,
        insights_gained || null
      ]
    )

    // Update daily session
    const today = moment().format('YYYY-MM-DD')
    await db.run(
      `
            INSERT OR REPLACE INTO daily_sessions (
                session_date, session_type, duration_minutes, quests_completed, focus_score
            ) VALUES (
                ?, 
                'coding',
                COALESCE((SELECT duration_minutes FROM daily_sessions WHERE session_date = ?), 0) + ?,
                COALESCE((SELECT quests_completed FROM daily_sessions WHERE session_date = ?), 0) + 1,
                ?
            )
        `,
      [today, today, time_to_complete_minutes, today, scores.quality || 8]
    )

    // Award skill points based on quest type and performance
    let skillCategory =
      quest_type === 'coding_exercise'
        ? 'programming'
        : quest_type === 'theory_quiz'
        ? 'theory'
        : 'applications'
    let basePoints =
      status === 'mastered' ? 15 : status === 'completed' ? 10 : 5

    await db.run(
      `
            INSERT INTO skill_points (
                category, points_earned, reason, source_type, related_quest_id, earned_at
            ) VALUES (?, ?, ?, 'quest_completion', ?, datetime('now'))
        `,
      [skillCategory, basePoints, `Completed quest: ${quest_title}`, quest_id]
    )

    const newQuest = await db.get(
      `
            SELECT * FROM quest_completions WHERE id = ?
        `,
      [result.lastID]
    )

    res.status(201).json({
      success: true,
      message: 'Quest completed successfully!',
      data: newQuest
    })
  })
)

// ==========================================
// SESSION MANAGEMENT ENDPOINTS
// ==========================================

// GET /api/v1/learning/sessions - Get daily learning sessions
router.get(
  '/sessions',
  [
    query('date')
      .optional()
      .isISO8601()
      .withMessage('Date must be in YYYY-MM-DD format'),
    query('type')
      .optional()
      .isIn(['math', 'coding', 'visual_projects', 'real_applications']),
    query('limit').optional().isInt({ min: 1, max: 100 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { date, type, limit = 30 } = req.query

    let whereClause = '1=1'
    let params = []

    if (date) {
      whereClause += ' AND session_date = ?'
      params.push(date)
    }

    if (type) {
      whereClause += ' AND session_type = ?'
      params.push(type)
    }

    const sessions = await db.all(
      `
            SELECT * FROM daily_sessions 
            WHERE ${whereClause}
            ORDER BY session_date DESC, id DESC
            LIMIT ?
        `,
      [...params, limit]
    )

    // Get today's session summary
    const today = moment().format('YYYY-MM-DD')
    const todayStats = await db.get(
      `
            SELECT 
                COUNT(*) as session_count,
                SUM(duration_minutes) as total_time,
                AVG(focus_score) as avg_focus,
                SUM(lessons_completed) as lessons_completed,
                SUM(quests_completed) as quests_completed
            FROM daily_sessions 
            WHERE session_date = ?
        `,
      [today]
    )

    // Get session type recommendation for tomorrow
    const recentSessions = await db.all(`
            SELECT session_type, COUNT(*) as count
            FROM daily_sessions 
            WHERE session_date >= date('now', '-7 days')
            GROUP BY session_type
            ORDER BY count ASC
        `)

    const sessionTypes = [
      'math',
      'coding',
      'visual_projects',
      'real_applications'
    ]
    const leastUsedType =
      recentSessions.length > 0
        ? recentSessions[0].session_type
        : sessionTypes[0]

    res.json({
      success: true,
      data: {
        sessions,
        todayStats: todayStats || {
          session_count: 0,
          total_time: 0,
          avg_focus: null,
          lessons_completed: 0,
          quests_completed: 0
        },
        recommendations: {
          nextSessionType: leastUsedType,
          sessionBalance: recentSessions
        }
      }
    })
  })
)

// POST /api/v1/learning/sessions - Start new learning session
router.post(
  '/sessions',
  [
    body('session_type')
      .isIn(['math', 'coding', 'visual_projects', 'real_applications'])
      .withMessage('Invalid session type'),
    body('planned_duration').optional().isInt({ min: 5, max: 180 }),
    body('goals').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { session_type, planned_duration = 25, goals } = req.body
    const today = moment().format('YYYY-MM-DD')

    // Create or update today's session
    const result = await db.run(
      `
            INSERT OR REPLACE INTO daily_sessions (
                session_date, session_type, planned_duration_minutes, goals, created_at
            ) VALUES (?, ?, ?, ?, datetime('now'))
        `,
      [today, session_type, planned_duration, goals || null]
    )

    const session = await db.get(
      `
            SELECT * FROM daily_sessions WHERE id = ?
        `,
      [result.lastID]
    )

    res.status(201).json({
      success: true,
      message: 'Learning session started!',
      data: session
    })
  })
)

// ==========================================
// SPACED REPETITION ENDPOINTS
// ==========================================

// GET /api/v1/learning/spaced-repetition - Get items due for review
router.get(
  '/spaced-repetition',
  asyncErrorHandler(async (req, res) => {
    const today = moment().format('YYYY-MM-DD')

    const reviewItems = await db.all(
      `
        SELECT 
            sr.*,
            lp.lesson_title,
            lp.lesson_type,
            lp.phase,
            lp.week
        FROM spaced_repetition sr
        LEFT JOIN learning_progress lp ON sr.concept_id = lp.lesson_id
        WHERE sr.next_review_date <= ?
        ORDER BY sr.next_review_date ASC, sr.difficulty_factor DESC
        LIMIT 20
    `,
      [today]
    )

    const upcomingReviews = await db.all(
      `
        SELECT 
            DATE(next_review_date) as review_date,
            COUNT(*) as items_count
        FROM spaced_repetition 
        WHERE next_review_date > ?
        GROUP BY DATE(next_review_date)
        ORDER BY next_review_date ASC
        LIMIT 7
    `,
      [today]
    )

    res.json({
      success: true,
      data: {
        reviewItems,
        upcomingReviews,
        todayCount: reviewItems.length
      }
    })
  })
)

// POST /api/v1/learning/spaced-repetition/:id/review - Record spaced repetition review
router.post(
  '/spaced-repetition/:id/review',
  [
    param('id').notEmpty().withMessage('Concept ID is required'),
    body('quality_score')
      .isInt({ min: 0, max: 5 })
      .withMessage('Quality score must be 0-5'),
    body('review_notes').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const { quality_score, review_notes } = req.body

    // Get current spaced repetition data
    const current = await db.get(
      `
            SELECT * FROM spaced_repetition WHERE concept_id = ?
        `,
      [id]
    )

    if (!current) {
      return res.status(404).json({
        success: false,
        message: 'Concept not found in spaced repetition system'
      })
    }

    // Calculate next review date using SM-2 algorithm
    let { difficulty_factor, interval_days, repetition_count } = current

    if (quality_score >= 3) {
      // Correct response
      if (repetition_count === 0) {
        interval_days = 1
      } else if (repetition_count === 1) {
        interval_days = 6
      } else {
        interval_days = Math.round(interval_days * difficulty_factor)
      }
      repetition_count += 1
    } else {
      // Incorrect response - reset
      repetition_count = 0
      interval_days = 1
    }

    // Update difficulty factor
    difficulty_factor = Math.max(
      1.3,
      difficulty_factor +
        (0.1 - (5 - quality_score) * (0.08 + (5 - quality_score) * 0.02))
    )

    const nextReviewDate = moment()
      .add(interval_days, 'days')
      .format('YYYY-MM-DD')

    // Update spaced repetition record
    await db.run(
      `
            UPDATE spaced_repetition 
            SET difficulty_factor = ?,
                interval_days = ?,
                repetition_count = ?,
                next_review_date = ?,
                last_reviewed_at = datetime('now'),
                quality_score = ?
            WHERE concept_id = ?
        `,
      [
        difficulty_factor,
        interval_days,
        repetition_count,
        nextReviewDate,
        quality_score,
        id
      ]
    )

    const updated = await db.get(
      `
            SELECT * FROM spaced_repetition WHERE concept_id = ?
        `,
      [id]
    )

    res.json({
      success: true,
      message: 'Review recorded successfully',
      data: {
        updated,
        nextReviewDate,
        intervalDays: interval_days
      }
    })
  })
)

// ==========================================
// ANALYTICS AND INSIGHTS ENDPOINTS
// ==========================================

// GET /api/v1/learning/analytics - Get learning analytics
router.get(
  '/analytics',
  asyncErrorHandler(async (req, res) => {
    const { timeframe = '30' } = req.query
    const days = parseInt(timeframe)
    const startDate = moment().subtract(days, 'days').format('YYYY-MM-DD')

    // Daily activity over time period
    const dailyActivity = await db.all(
      `
        SELECT 
            session_date,
            SUM(duration_minutes) as total_time,
            AVG(focus_score) as avg_focus,
            SUM(lessons_completed) as lessons,
            SUM(quests_completed) as quests
        FROM daily_sessions 
        WHERE session_date >= ?
        GROUP BY session_date
        ORDER BY session_date
    `,
      [startDate]
    )

    // Learning velocity (lessons per week)
    const weeklyProgress = await db.all(`
        SELECT 
            strftime('%Y-%W', completed_at) as week,
            COUNT(*) as lessons_completed,
            AVG(mastery_score) as avg_mastery,
            SUM(time_spent_minutes) as total_time
        FROM learning_progress 
        WHERE completed_at >= date('now', '-8 weeks')
            AND status IN ('completed', 'mastered')
        GROUP BY strftime('%Y-%W', completed_at)
        ORDER BY week
    `)

    // Skill distribution
    const skillDistribution = await db.all(
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

    // Difficulty patterns
    const difficultyAnalysis = await db.all(
      `
        SELECT 
            difficulty_rating,
            COUNT(*) as count,
            AVG(time_spent_minutes) as avg_time,
            AVG(mastery_score) as avg_mastery
        FROM learning_progress 
        WHERE difficulty_rating IS NOT NULL
            AND completed_at >= ?
        GROUP BY difficulty_rating
        ORDER BY difficulty_rating
    `,
      [startDate]
    )

    // Focus trends
    const focusAnalysis = await db.all(
      `
        SELECT 
            session_type,
            AVG(focus_score) as avg_focus,
            AVG(duration_minutes) as avg_duration,
            COUNT(*) as session_count
        FROM daily_sessions 
        WHERE session_date >= ?
            AND focus_score IS NOT NULL
        GROUP BY session_type
    `,
      [startDate]
    )

    res.json({
      success: true,
      data: {
        timeframe: `${days} days`,
        dailyActivity,
        weeklyProgress,
        skillDistribution,
        difficultyAnalysis,
        focusAnalysis,
        insights: {
          totalStudyTime: dailyActivity.reduce(
            (sum, day) => sum + (day.total_time || 0),
            0
          ),
          averageFocus:
            dailyActivity.filter(d => d.avg_focus).length > 0
              ? dailyActivity
                  .filter(d => d.avg_focus)
                  .reduce((sum, d) => sum + d.avg_focus, 0) /
                dailyActivity.filter(d => d.avg_focus).length
              : null,
          mostProductiveType:
            focusAnalysis.length > 0
              ? focusAnalysis.reduce(
                  (max, curr) => (curr.avg_focus > max.avg_focus ? curr : max),
                  focusAnalysis[0]
                ).session_type
              : null
        }
      }
    })
  })
)

// GET /api/v1/learning/streak - Get current learning streak
router.get(
  '/streak',
  asyncErrorHandler(async (req, res) => {
    const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

    // Calculate current streak
    const recentSessions = await db.all(`
        SELECT DISTINCT session_date 
        FROM daily_sessions 
        WHERE session_date <= date('now')
        ORDER BY session_date DESC
        LIMIT 100
    `)

    let currentStreak = 0
    let expectedDate = moment()

    for (const session of recentSessions) {
      if (moment(session.session_date).isSame(expectedDate, 'day')) {
        currentStreak++
        expectedDate.subtract(1, 'day')
      } else {
        break
      }
    }

    // Update profile if streak changed
    if (currentStreak !== profile.current_streak_days) {
      await db.run(
        `
            UPDATE user_profile 
            SET current_streak_days = ?,
                longest_streak_days = MAX(longest_streak_days, ?),
                updated_at = datetime('now')
            WHERE id = 1
        `,
        [currentStreak, currentStreak]
      )
    }

    // Get streak history
    const streakHistory = await db.all(`
        SELECT 
            session_date,
            SUM(duration_minutes) as daily_time,
            COUNT(*) as activities
        FROM daily_sessions 
        WHERE session_date >= date('now', '-30 days')
        GROUP BY session_date
        ORDER BY session_date
    `)

    res.json({
      success: true,
      data: {
        currentStreak,
        longestStreak: Math.max(profile.longest_streak_days, currentStreak),
        streakHistory,
        streakGoal: profile.daily_goal_minutes || 60
      }
    })
  })
)

// GET /api/v1/learning/next-lesson - Get next lesson recommendation
router.get(
  '/next-lesson',
  asyncErrorHandler(async (req, res) => {
    const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

    // Find next incomplete lesson in current phase/week
    const nextLesson = await db.get(
      `
        SELECT * FROM learning_progress 
        WHERE phase = ? AND week = ? AND status = 'not_started'
        ORDER BY lesson_id
        LIMIT 1
    `,
      [profile.current_phase, profile.current_week]
    )

    if (nextLesson) {
      res.json({
        success: true,
        data: {
          lesson: nextLesson,
          recommendation: 'Continue current week',
          progress: `Phase ${profile.current_phase}, Week ${profile.current_week}`
        }
      })
      return
    }

    // Check if current week is complete
    const currentWeekComplete = await db.get(
      `
        SELECT COUNT(*) as incomplete
        FROM learning_progress 
        WHERE phase = ? AND week = ? AND status = 'not_started'
    `,
      [profile.current_phase, profile.current_week]
    )

    if (currentWeekComplete.incomplete === 0) {
      // Move to next week or phase
      const nextWeekLesson = await db.get(
        `
            SELECT * FROM learning_progress 
            WHERE phase = ? AND week = ? AND status = 'not_started'
            ORDER BY lesson_id
            LIMIT 1
        `,
        [profile.current_phase, profile.current_week + 1]
      )

      if (nextWeekLesson) {
        // Update profile to next week
        await db.run(`
                UPDATE user_profile 
                SET current_week = current_week + 1,
                    updated_at = datetime('now')
                WHERE id = 1
            `)

        res.json({
          success: true,
          data: {
            lesson: nextWeekLesson,
            recommendation: 'Starting next week!',
            progress: `Phase ${profile.current_phase}, Week ${
              profile.current_week + 1
            }`
          }
        })
        return
      }
    }

    // Recommend review or next phase
    const reviewItems = await db.all(`
        SELECT * FROM learning_progress 
        WHERE status = 'completed' AND mastery_score < 0.8
        ORDER BY completed_at DESC
        LIMIT 5
    `)

    res.json({
      success: true,
      data: {
        lesson: null,
        recommendation:
          reviewItems.length > 0 ? 'Review previous lessons' : 'All caught up!',
        reviewSuggestions: reviewItems,
        progress: `Phase ${profile.current_phase}, Week ${profile.current_week}`
      }
    })
  })
)

module.exports = router
