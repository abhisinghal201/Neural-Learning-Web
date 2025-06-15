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

// GET /api/v1/learning/progress - Get overall learning progress
router.get(
  '/progress',
  asyncErrorHandler(async (req, res) => {
    try {
      // Get user profile and current progress
      const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

      if (!profile) {
        // Create default profile if none exists
        await db.run(`
                INSERT INTO user_profile (
                    id, name, email, learning_goal, current_phase, current_week,
                    total_study_time_minutes, streak_days, max_streak,
                    created_at, updated_at
                ) VALUES (1, 'Neural Explorer', 'explorer@neural-odyssey.local', 
                         'Become AI/ML Expert', 1, 1, 0, 0, 0, 
                         datetime('now'), datetime('now'))
            `)

        const newProfile = await db.get(
          'SELECT * FROM user_profile WHERE id = 1'
        )
        profile = newProfile
      }

      // Get overall progress statistics
      const overallStats = await db.get(`
            SELECT 
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_lessons,
                AVG(CASE WHEN status IN ('completed', 'mastered') THEN understanding_score END) as avg_understanding,
                SUM(time_spent_minutes) as total_time_minutes
            FROM learning_progress
        `)

      // Get progress by phase
      const phaseProgress = await db.all(`
            SELECT 
                phase,
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                AVG(CASE WHEN status IN ('completed', 'mastered') THEN understanding_score END) as avg_understanding
            FROM learning_progress
            GROUP BY phase
            ORDER BY phase
        `)

      // Get recent quest completions
      const recentQuests = await db.all(`
            SELECT quest_id, quest_title, status, code_quality_score, 
                   time_to_complete_minutes, completed_at
            FROM quest_completions
            WHERE status IN ('completed', 'mastered')
            ORDER BY completed_at DESC
            LIMIT 5
        `)

      // Get current week's sessions
      const weekStart = moment().startOf('week').format('YYYY-MM-DD')
      const weekEnd = moment().endOf('week').format('YYYY-MM-DD')

      const weekSessions = await db.all(
        `
            SELECT date, study_time_minutes, quests_completed, 
                   lessons_completed, focus_score
            FROM daily_sessions
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        `,
        [weekStart, weekEnd]
      )

      // Calculate completion percentage
      const completionPercentage =
        overallStats.total_lessons > 0
          ? Math.round(
              (overallStats.completed_lessons / overallStats.total_lessons) *
                100
            )
          : 0

      // Build response
      const response = {
        success: true,
        data: {
          profile: {
            name: profile.name,
            currentPhase: profile.current_phase,
            currentWeek: profile.current_week,
            learningGoal: profile.learning_goal,
            totalStudyTime: profile.total_study_time_minutes,
            streakDays: profile.streak_days,
            maxStreak: profile.max_streak
          },
          overall: {
            totalLessons: overallStats.total_lessons || 0,
            completedLessons: overallStats.completed_lessons || 0,
            masteredLessons: overallStats.mastered_lessons || 0,
            inProgressLessons: overallStats.in_progress_lessons || 0,
            completionPercentage,
            averageUnderstanding: Math.round(
              overallStats.avg_understanding || 0
            ),
            totalTimeMinutes: overallStats.total_time_minutes || 0
          },
          phases: phaseProgress.map(phase => ({
            phase: phase.phase,
            totalLessons: phase.total_lessons,
            completedLessons: phase.completed_lessons,
            masteredLessons: phase.mastered_lessons,
            completionPercentage:
              phase.total_lessons > 0
                ? Math.round(
                    (phase.completed_lessons / phase.total_lessons) * 100
                  )
                : 0,
            averageUnderstanding: Math.round(phase.avg_understanding || 0)
          })),
          recentQuests: recentQuests.map(quest => ({
            id: quest.quest_id,
            title: quest.quest_title,
            status: quest.status,
            qualityScore: quest.code_quality_score,
            timeToComplete: quest.time_to_complete_minutes,
            completedAt: quest.completed_at
          })),
          weekSessions: weekSessions.map(session => ({
            date: session.date,
            studyTime: session.study_time_minutes,
            questsCompleted: session.quests_completed,
            lessonsCompleted: session.lessons_completed,
            focusScore: session.focus_score
          }))
        }
      }

      res.json(response)
    } catch (error) {
      console.error('❌ Error getting learning progress:', error)
      throw error
    }
  })
)

// GET /api/v1/learning/progress/phase/:phase - Get phase-specific progress
router.get(
  '/progress/phase/:phase',
  [param('phase').isInt({ min: 1, max: 4 }).withMessage('Phase must be 1-4')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase } = req.params

    // Get phase progress with detailed lesson breakdown
    const lessons = await db.all(
      `
        SELECT 
            lesson_id, lesson_title, lesson_type, phase, week,
            status, understanding_score, confidence_level,
            time_spent_minutes, attempts_count, last_accessed,
            notes, difficulty_rating, created_at, updated_at
        FROM learning_progress
        WHERE phase = ?
        ORDER BY week, lesson_id
    `,
      [phase]
    )

    // Get phase statistics
    const phaseStats = await db.get(
      `
        SELECT 
            COUNT(*) as total_lessons,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_lessons,
            COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
            COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_lessons,
            AVG(understanding_score) as avg_understanding,
            AVG(confidence_level) as avg_confidence,
            SUM(time_spent_minutes) as total_time_minutes,
            MAX(week) as max_week
        FROM learning_progress
        WHERE phase = ?
    `,
      [phase]
    )

    // Group lessons by week
    const lessonsByWeek = {}
    lessons.forEach(lesson => {
      const week = lesson.week
      if (!lessonsByWeek[week]) {
        lessonsByWeek[week] = []
      }
      lessonsByWeek[week].push({
        id: lesson.lesson_id,
        title: lesson.lesson_title,
        type: lesson.lesson_type,
        status: lesson.status,
        understanding: lesson.understanding_score,
        confidence: lesson.confidence_level,
        timeSpent: lesson.time_spent_minutes,
        attempts: lesson.attempts_count,
        lastAccessed: lesson.last_accessed,
        notes: lesson.notes,
        difficulty: lesson.difficulty_rating,
        updatedAt: lesson.updated_at
      })
    })

    res.json({
      success: true,
      data: {
        phase: parseInt(phase),
        statistics: {
          totalLessons: phaseStats.total_lessons || 0,
          completedLessons: phaseStats.completed_lessons || 0,
          masteredLessons: phaseStats.mastered_lessons || 0,
          inProgressLessons: phaseStats.in_progress_lessons || 0,
          completionPercentage:
            phaseStats.total_lessons > 0
              ? Math.round(
                  (phaseStats.completed_lessons / phaseStats.total_lessons) *
                    100
                )
              : 0,
          averageUnderstanding: Math.round(phaseStats.avg_understanding || 0),
          averageConfidence: Math.round(phaseStats.avg_confidence || 0),
          totalTimeMinutes: phaseStats.total_time_minutes || 0,
          totalWeeks: phaseStats.max_week || 0
        },
        lessonsByWeek
      }
    })
  })
)

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
      .isIn(['theory', 'practical', 'project', 'assessment']),
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase, week, status, type, limit = 50, offset = 0 } = req.query

    let whereClause = []
    let params = []

    if (phase) {
      whereClause.push('phase = ?')
      params.push(phase)
    }

    if (week) {
      whereClause.push('week = ?')
      params.push(week)
    }

    if (status) {
      whereClause.push('status = ?')
      params.push(status)
    }

    if (type) {
      whereClause.push('lesson_type = ?')
      params.push(type)
    }

    const whereSQL =
      whereClause.length > 0 ? `WHERE ${whereClause.join(' AND ')}` : ''

    // Get total count
    const countQuery = `SELECT COUNT(*) as total FROM learning_progress ${whereSQL}`
    const { total } = await db.get(countQuery, params)

    // Get lessons with pagination
    const lessonsQuery = `
        SELECT 
            lesson_id, lesson_title, lesson_type, phase, week,
            status, understanding_score, confidence_level,
            time_spent_minutes, attempts_count, last_accessed,
            notes, difficulty_rating, created_at, updated_at
        FROM learning_progress
        ${whereSQL}
        ORDER BY phase, week, lesson_id
        LIMIT ? OFFSET ?
    `

    const lessons = await db.all(lessonsQuery, [...params, limit, offset])

    res.json({
      success: true,
      data: {
        lessons: lessons.map(lesson => ({
          id: lesson.lesson_id,
          title: lesson.lesson_title,
          type: lesson.lesson_type,
          phase: lesson.phase,
          week: lesson.week,
          status: lesson.status,
          understanding: lesson.understanding_score,
          confidence: lesson.confidence_level,
          timeSpent: lesson.time_spent_minutes,
          attempts: lesson.attempts_count,
          lastAccessed: lesson.last_accessed,
          notes: lesson.notes,
          difficulty: lesson.difficulty_rating,
          createdAt: lesson.created_at,
          updatedAt: lesson.updated_at
        })),
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

// GET /api/v1/learning/lessons/:id - Get specific lesson details
router.get(
  '/lessons/:id',
  [param('id').notEmpty().withMessage('Lesson ID is required')],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params

    // Get lesson progress
    const lesson = await db.get(
      `
        SELECT 
            lesson_id, lesson_title, lesson_type, phase, week,
            status, understanding_score, confidence_level,
            time_spent_minutes, attempts_count, last_accessed,
            notes, difficulty_rating, created_at, updated_at
        FROM learning_progress
        WHERE lesson_id = ?
    `,
      [id]
    )

    if (!lesson) {
      throw new ProgressNotFoundError(id)
    }

    // Get related quest completions
    const quests = await db.all(
      `
        SELECT quest_id, quest_title, status, attempts_count,
               time_to_complete_minutes, code_quality_score,
               completed_at
        FROM quest_completions
        WHERE related_lesson_id = ?
        ORDER BY completed_at DESC
    `,
      [id]
    )

    // Get spaced repetition items for this lesson
    const repetitionItems = await db.all(
      `
        SELECT item_id, item_type, ease_factor, interval_days,
               next_review_date, review_count, last_reviewed
        FROM spaced_repetition
        WHERE lesson_id = ?
        ORDER BY next_review_date
    `,
      [id]
    )

    // Try to load lesson content from file system
    let contentPath = null
    let contentExists = false

    try {
      const contentBasePath = path.join(
        __dirname,
        `../../content/phases/phase-${lesson.phase}-foundations/week-${lesson.week}-*/`
      )
      const possiblePaths = [
        path.join(
          __dirname,
          `../../content/phases/phase-${lesson.phase}-foundations/week-${lesson.week}/lesson.md`
        ),
        path.join(
          __dirname,
          `../../content/phases/phase-${lesson.phase}-foundations/week-${lesson.week}/exercises.py`
        ),
        path.join(
          __dirname,
          `../../content/phases/phase-${lesson.phase}-foundations/week-${lesson.week}/resources.json`
        )
      ]

      for (const possiblePath of possiblePaths) {
        if (fs.existsSync(possiblePath)) {
          contentPath = possiblePath
          contentExists = true
          break
        }
      }
    } catch (error) {
      console.warn('⚠️ Could not locate lesson content files:', error.message)
    }

    res.json({
      success: true,
      data: {
        lesson: {
          id: lesson.lesson_id,
          title: lesson.lesson_title,
          type: lesson.lesson_type,
          phase: lesson.phase,
          week: lesson.week,
          status: lesson.status,
          understanding: lesson.understanding_score,
          confidence: lesson.confidence_level,
          timeSpent: lesson.time_spent_minutes,
          attempts: lesson.attempts_count,
          lastAccessed: lesson.last_accessed,
          notes: lesson.notes,
          difficulty: lesson.difficulty_rating,
          createdAt: lesson.created_at,
          updatedAt: lesson.updated_at,
          contentExists
        },
        quests: quests.map(quest => ({
          id: quest.quest_id,
          title: quest.quest_title,
          status: quest.status,
          attempts: quest.attempts_count,
          timeToComplete: quest.time_to_complete_minutes,
          qualityScore: quest.code_quality_score,
          completedAt: quest.completed_at
        })),
        repetitionItems: repetitionItems.map(item => ({
          id: item.item_id,
          type: item.item_type,
          easeFactor: item.ease_factor,
          intervalDays: item.interval_days,
          nextReviewDate: item.next_review_date,
          reviewCount: item.review_count,
          lastReviewed: item.last_reviewed
        }))
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
    body('understanding').optional().isInt({ min: 1, max: 5 }),
    body('confidence').optional().isInt({ min: 1, max: 5 }),
    body('timeSpent').optional().isInt({ min: 0 }),
    body('notes').optional().isString().isLength({ max: 1000 }),
    body('difficulty').optional().isInt({ min: 1, max: 5 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const { status, understanding, confidence, timeSpent, notes, difficulty } =
      req.body

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

    // Build update query
    const updates = []
    const params = []

    if (status !== undefined) {
      updates.push('status = ?')
      params.push(status)
    }

    if (understanding !== undefined) {
      updates.push('understanding_score = ?')
      params.push(understanding)
    }

    if (confidence !== undefined) {
      updates.push('confidence_level = ?')
      params.push(confidence)
    }

    if (timeSpent !== undefined) {
      updates.push('time_spent_minutes = time_spent_minutes + ?')
      params.push(timeSpent)
    }

    if (notes !== undefined) {
      updates.push('notes = ?')
      params.push(notes)
    }

    if (difficulty !== undefined) {
      updates.push('difficulty_rating = ?')
      params.push(difficulty)
    }

    // Always update last_accessed and updated_at
    updates.push('last_accessed = datetime("now")')
    updates.push('updated_at = datetime("now")')
    updates.push('attempts_count = attempts_count + 1')

    params.push(id)

    const updateQuery = `
        UPDATE learning_progress 
        SET ${updates.join(', ')}
        WHERE lesson_id = ?
    `

    await db.run(updateQuery, params)

    // Get updated lesson
    const updatedLesson = await db.get(
      `
        SELECT * FROM learning_progress WHERE lesson_id = ?
    `,
      [id]
    )

    // Update user profile if lesson was completed
    if (status === 'completed' || status === 'mastered') {
      await db.run(
        `
            UPDATE user_profile 
            SET total_study_time_minutes = total_study_time_minutes + ?,
                updated_at = datetime('now')
            WHERE id = 1
        `,
        [timeSpent || 0]
      )
    }

    res.json({
      success: true,
      message: 'Lesson progress updated successfully',
      data: {
        lesson: {
          id: updatedLesson.lesson_id,
          title: updatedLesson.lesson_title,
          type: updatedLesson.lesson_type,
          phase: updatedLesson.phase,
          week: updatedLesson.week,
          status: updatedLesson.status,
          understanding: updatedLesson.understanding_score,
          confidence: updatedLesson.confidence_level,
          timeSpent: updatedLesson.time_spent_minutes,
          attempts: updatedLesson.attempts_count,
          lastAccessed: updatedLesson.last_accessed,
          notes: updatedLesson.notes,
          difficulty: updatedLesson.difficulty_rating,
          updatedAt: updatedLesson.updated_at
        }
      }
    })
  })
)

// POST /api/v1/learning/lessons/:id/complete - Mark lesson as completed
router.post(
  '/lessons/:id/complete',
  [
    param('id').notEmpty().withMessage('Lesson ID is required'),
    body('understanding')
      .isInt({ min: 1, max: 5 })
      .withMessage('Understanding score (1-5) is required'),
    body('confidence')
      .isInt({ min: 1, max: 5 })
      .withMessage('Confidence level (1-5) is required'),
    body('timeSpent')
      .isInt({ min: 1 })
      .withMessage('Time spent (minutes) is required'),
    body('notes').optional().isString().isLength({ max: 1000 }),
    body('difficulty').optional().isInt({ min: 1, max: 5 }),
    body('mastered').optional().isBoolean()
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const {
      understanding,
      confidence,
      timeSpent,
      notes,
      difficulty,
      mastered = false
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

    const finalStatus = mastered ? 'mastered' : 'completed'

    // Update lesson progress
    await db.run(
      `
        UPDATE learning_progress 
        SET status = ?,
            understanding_score = ?,
            confidence_level = ?,
            time_spent_minutes = time_spent_minutes + ?,
            notes = ?,
            difficulty_rating = ?,
            last_accessed = datetime('now'),
            updated_at = datetime('now'),
            attempts_count = attempts_count + 1
        WHERE lesson_id = ?
    `,
      [
        finalStatus,
        understanding,
        confidence,
        timeSpent,
        notes || '',
        difficulty || 3,
        id
      ]
    )

    // Update user profile
    await db.run(
      `
        UPDATE user_profile 
        SET total_study_time_minutes = total_study_time_minutes + ?,
            updated_at = datetime('now')
        WHERE id = 1
    `,
      [timeSpent]
    )

    // Create spaced repetition items if lesson was mastered
    if (mastered) {
      await db.run(
        `
            INSERT OR REPLACE INTO spaced_repetition (
                lesson_id, item_id, item_type, ease_factor, interval_days,
                next_review_date, review_count, last_reviewed, created_at
            ) VALUES (?, ?, 'lesson_concept', 2.5, 1, 
                     date('now', '+1 day'), 0, datetime('now'), datetime('now'))
        `,
        [id, `${id}_concept`]
      )
    }

    res.json({
      success: true,
      message: `Lesson ${mastered ? 'mastered' : 'completed'} successfully!`,
      data: {
        lessonId: id,
        status: finalStatus,
        understanding,
        confidence,
        timeSpent,
        totalTimeSpent: existingLesson.time_spent_minutes + timeSpent
      }
    })
  })
)

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
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { phase, week, status, type, limit = 50, offset = 0 } = req.query

    let whereClause = []
    let params = []

    if (phase) {
      whereClause.push('phase = ?')
      params.push(phase)
    }

    if (week) {
      whereClause.push('week = ?')
      params.push(week)
    }

    if (status) {
      whereClause.push('status = ?')
      params.push(status)
    }

    if (type) {
      whereClause.push('quest_type = ?')
      params.push(type)
    }

    const whereSQL =
      whereClause.length > 0 ? `WHERE ${whereClause.join(' AND ')}` : ''

    // Get total count
    const countQuery = `SELECT COUNT(*) as total FROM quest_completions ${whereSQL}`
    const { total } = await db.get(countQuery, params)

    // Get quests with pagination
    const questsQuery = `
        SELECT 
            id, quest_id, quest_title, quest_type, phase, week,
            related_lesson_id, status, attempts_count, time_to_complete_minutes,
            user_code, final_solution, test_results,
            code_quality_score, creativity_score, efficiency_score,
            learning_notes, challenges_faced, insights_gained,
            completed_at, created_at, updated_at
        FROM quest_completions
        ${whereSQL}
        ORDER BY phase, week, quest_id, completed_at DESC
        LIMIT ? OFFSET ?
    `

    const quests = await db.all(questsQuery, [...params, limit, offset])

    res.json({
      success: true,
      data: {
        quests: quests.map(quest => ({
          id: quest.id,
          questId: quest.quest_id,
          title: quest.quest_title,
          type: quest.quest_type,
          phase: quest.phase,
          week: quest.week,
          relatedLessonId: quest.related_lesson_id,
          status: quest.status,
          attempts: quest.attempts_count,
          timeToComplete: quest.time_to_complete_minutes,
          scores: {
            quality: quest.code_quality_score,
            creativity: quest.creativity_score,
            efficiency: quest.efficiency_score
          },
          learningNotes: quest.learning_notes,
          challengesFaced: quest.challenges_faced,
          insightsGained: quest.insights_gained,
          completedAt: quest.completed_at,
          createdAt: quest.created_at,
          updatedAt: quest.updated_at
        })),
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

// POST /api/v1/learning/quests - Create new quest completion
router.post(
  '/quests',
  [
    body('questId').notEmpty().withMessage('Quest ID is required'),
    body('questTitle').notEmpty().withMessage('Quest title is required'),
    body('questType').isIn([
      'coding_exercise',
      'implementation_project',
      'theory_quiz',
      'practical_application'
    ]),
    body('phase').isInt({ min: 1, max: 4 }).withMessage('Phase must be 1-4'),
    body('week').isInt({ min: 1 }).withMessage('Week must be positive integer'),
    body('status').isIn(['attempted', 'completed', 'mastered']),
    body('timeToComplete')
      .isInt({ min: 1 })
      .withMessage('Time to complete is required'),
    body('userCode').optional().isString(),
    body('finalSolution').optional().isString(),
    body('testResults').optional().isString(),
    body('relatedLessonId').optional().isString(),
    body('scores').optional().isObject(),
    body('learningNotes').optional().isString().isLength({ max: 2000 }),
    body('challengesFaced').optional().isString().isLength({ max: 1000 }),
    body('insightsGained').optional().isString().isLength({ max: 1000 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      questId,
      questTitle,
      questType,
      phase,
      week,
      status,
      timeToComplete,
      userCode,
      finalSolution,
      testResults,
      relatedLessonId,
      scores = {},
      learningNotes,
      challengesFaced,
      insightsGained
    } = req.body

    // Insert quest completion
    const result = await db.run(
      `
        INSERT INTO quest_completions (
            quest_id, quest_title, quest_type, phase, week,
            related_lesson_id, status, attempts_count, time_to_complete_minutes,
            user_code, final_solution, test_results,
            code_quality_score, creativity_score, efficiency_score,
            learning_notes, challenges_faced, insights_gained,
            completed_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                 datetime('now'), datetime('now'), datetime('now'))
    `,
      [
        questId,
        questTitle,
        questType,
        phase,
        week,
        relatedLessonId || null,
        status,
        timeToComplete,
        userCode || null,
        finalSolution || null,
        testResults || null,
        scores.quality || null,
        scores.creativity || null,
        scores.efficiency || null,
        learningNotes || null,
        challengesFaced || null,
        insightsGained || null
      ]
    )

    // Update daily session
    const today = moment().format('YYYY-MM-DD')
    await db.run(
      `
        INSERT OR REPLACE INTO daily_sessions (
            date, study_time_minutes, quests_completed, lessons_completed,
            focus_score, notes, created_at, updated_at
        ) VALUES (
            ?, 
            COALESCE((SELECT study_time_minutes FROM daily_sessions WHERE date = ?), 0) + ?,
            COALESCE((SELECT quests_completed FROM daily_sessions WHERE date = ?), 0) + 1,
            COALESCE((SELECT lessons_completed FROM daily_sessions WHERE date = ?), 0),
            COALESCE((SELECT focus_score FROM daily_sessions WHERE date = ?), 75),
            COALESCE((SELECT notes FROM daily_sessions WHERE date = ?), ''),
            COALESCE((SELECT created_at FROM daily_sessions WHERE date = ?), datetime('now')),
            datetime('now')
        )
    `,
      [today, today, timeToComplete, today, today, today, today, today]
    )

    res.status(201).json({
      success: true,
      message: 'Quest completion recorded successfully!',
      data: {
        id: result.lastID,
        questId,
        status,
        timeToComplete
      }
    })
  })
)

// PUT /api/v1/learning/quests/:id - Update quest completion
router.put(
  '/quests/:id',
  [
    param('id').isInt().withMessage('Quest completion ID must be an integer'),
    body('status').optional().isIn(['attempted', 'completed', 'mastered']),
    body('userCode').optional().isString(),
    body('finalSolution').optional().isString(),
    body('testResults').optional().isString(),
    body('scores').optional().isObject(),
    body('learningNotes').optional().isString().isLength({ max: 2000 }),
    body('challengesFaced').optional().isString().isLength({ max: 1000 }),
    body('insightsGained').optional().isString().isLength({ max: 1000 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { id } = req.params
    const {
      status,
      userCode,
      finalSolution,
      testResults,
      scores = {},
      learningNotes,
      challengesFaced,
      insightsGained
    } = req.body

    // Check if quest completion exists
    const existingQuest = await db.get(
      `
        SELECT * FROM quest_completions WHERE id = ?
    `,
      [id]
    )

    if (!existingQuest) {
      throw new QuestNotFoundError(id)
    }

    // Build update query
    const updates = []
    const params = []

    if (status !== undefined) {
      updates.push('status = ?')
      params.push(status)
    }

    if (userCode !== undefined) {
      updates.push('user_code = ?')
      params.push(userCode)
    }

    if (finalSolution !== undefined) {
      updates.push('final_solution = ?')
      params.push(finalSolution)
    }

    if (testResults !== undefined) {
      updates.push('test_results = ?')
      params.push(testResults)
    }

    if (scores.quality !== undefined) {
      updates.push('code_quality_score = ?')
      params.push(scores.quality)
    }

    if (scores.creativity !== undefined) {
      updates.push('creativity_score = ?')
      params.push(scores.creativity)
    }

    if (scores.efficiency !== undefined) {
      updates.push('efficiency_score = ?')
      params.push(scores.efficiency)
    }

    if (learningNotes !== undefined) {
      updates.push('learning_notes = ?')
      params.push(learningNotes)
    }

    if (challengesFaced !== undefined) {
      updates.push('challenges_faced = ?')
      params.push(challengesFaced)
    }

    if (insightsGained !== undefined) {
      updates.push('insights_gained = ?')
      params.push(insightsGained)
    }

    // Always update attempts and timestamp
    updates.push('attempts_count = attempts_count + 1')
    updates.push('updated_at = datetime("now")')

    if (status === 'completed' || status === 'mastered') {
      updates.push('completed_at = datetime("now")')
    }

    params.push(id)

    const updateQuery = `
        UPDATE quest_completions 
        SET ${updates.join(', ')}
        WHERE id = ?
    `

    await db.run(updateQuery, params)

    // Get updated quest
    const updatedQuest = await db.get(
      `
        SELECT * FROM quest_completions WHERE id = ?
    `,
      [id]
    )

    res.json({
      success: true,
      message: 'Quest completion updated successfully',
      data: {
        id: updatedQuest.id,
        questId: updatedQuest.quest_id,
        status: updatedQuest.status,
        attempts: updatedQuest.attempts_count,
        updatedAt: updatedQuest.updated_at
      }
    })
  })
)

// GET /api/v1/learning/sessions - Get daily learning sessions
router.get(
  '/sessions',
  [
    query('startDate')
      .optional()
      .isISO8601()
      .withMessage('Start date must be valid ISO8601 date'),
    query('endDate')
      .optional()
      .isISO8601()
      .withMessage('End date must be valid ISO8601 date'),
    query('limit').optional().isInt({ min: 1, max: 365 }),
    query('offset').optional().isInt({ min: 0 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { startDate, endDate, limit = 30, offset = 0 } = req.query

    let whereClause = []
    let params = []

    if (startDate) {
      whereClause.push('date >= ?')
      params.push(moment(startDate).format('YYYY-MM-DD'))
    }

    if (endDate) {
      whereClause.push('date <= ?')
      params.push(moment(endDate).format('YYYY-MM-DD'))
    }

    const whereSQL =
      whereClause.length > 0 ? `WHERE ${whereClause.join(' AND ')}` : ''

    // Get total count
    const countQuery = `SELECT COUNT(*) as total FROM daily_sessions ${whereSQL}`
    const { total } = await db.get(countQuery, params)

    // Get sessions with pagination
    const sessionsQuery = `
        SELECT 
            date, study_time_minutes, quests_completed, lessons_completed,
            focus_score, notes, created_at, updated_at
        FROM daily_sessions
        ${whereSQL}
        ORDER BY date DESC
        LIMIT ? OFFSET ?
    `

    const sessions = await db.all(sessionsQuery, [...params, limit, offset])

    // Calculate statistics
    const stats = {
      totalStudyTime: 0,
      totalQuests: 0,
      totalLessons: 0,
      averageFocusScore: 0,
      streakDays: 0
    }

    if (sessions.length > 0) {
      stats.totalStudyTime = sessions.reduce(
        (sum, s) => sum + (s.study_time_minutes || 0),
        0
      )
      stats.totalQuests = sessions.reduce(
        (sum, s) => sum + (s.quests_completed || 0),
        0
      )
      stats.totalLessons = sessions.reduce(
        (sum, s) => sum + (s.lessons_completed || 0),
        0
      )
      stats.averageFocusScore = Math.round(
        sessions.reduce((sum, s) => sum + (s.focus_score || 0), 0) /
          sessions.length
      )

      // Calculate current streak
      const sortedSessions = sessions.sort(
        (a, b) => new Date(b.date) - new Date(a.date)
      )
      let streakCount = 0
      let currentDate = moment()

      for (const session of sortedSessions) {
        const sessionDate = moment(session.date)
        if (
          sessionDate.isSame(currentDate, 'day') ||
          sessionDate.isSame(currentDate.subtract(1, 'day'), 'day')
        ) {
          streakCount++
          currentDate = sessionDate
        } else {
          break
        }
      }

      stats.streakDays = streakCount
    }

    res.json({
      success: true,
      data: {
        sessions: sessions.map(session => ({
          date: session.date,
          studyTime: session.study_time_minutes,
          questsCompleted: session.quests_completed,
          lessonsCompleted: session.lessons_completed,
          focusScore: session.focus_score,
          notes: session.notes,
          createdAt: session.created_at,
          updatedAt: session.updated_at
        })),
        statistics: stats,
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

// POST /api/v1/learning/sessions - Start new learning session
router.post(
  '/sessions',
  [
    body('date')
      .optional()
      .isISO8601()
      .withMessage('Date must be valid ISO8601 date'),
    body('studyTime')
      .optional()
      .isInt({ min: 0 })
      .withMessage('Study time must be non-negative'),
    body('questsCompleted').optional().isInt({ min: 0 }),
    body('lessonsCompleted').optional().isInt({ min: 0 }),
    body('focusScore').optional().isInt({ min: 1, max: 100 }),
    body('notes').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      date = moment().format('YYYY-MM-DD'),
      studyTime = 0,
      questsCompleted = 0,
      lessonsCompleted = 0,
      focusScore = 75,
      notes = ''
    } = req.body

    // Insert or update daily session
    await db.run(
      `
        INSERT OR REPLACE INTO daily_sessions (
            date, study_time_minutes, quests_completed, lessons_completed,
            focus_score, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 
                 COALESCE((SELECT created_at FROM daily_sessions WHERE date = ?), datetime('now')),
                 datetime('now'))
    `,
      [
        date,
        studyTime,
        questsCompleted,
        lessonsCompleted,
        focusScore,
        notes,
        date
      ]
    )

    // Get the session
    const session = await db.get(
      `
        SELECT * FROM daily_sessions WHERE date = ?
    `,
      [date]
    )

    res.status(201).json({
      success: true,
      message: 'Learning session recorded successfully!',
      data: {
        date: session.date,
        studyTime: session.study_time_minutes,
        questsCompleted: session.quests_completed,
        lessonsCompleted: session.lessons_completed,
        focusScore: session.focus_score,
        notes: session.notes
      }
    })
  })
)

// PUT /api/v1/learning/sessions/:date - Update learning session
router.put(
  '/sessions/:date',
  [
    param('date').isISO8601().withMessage('Date must be valid ISO8601 date'),
    body('studyTime').optional().isInt({ min: 0 }),
    body('questsCompleted').optional().isInt({ min: 0 }),
    body('lessonsCompleted').optional().isInt({ min: 0 }),
    body('focusScore').optional().isInt({ min: 1, max: 100 }),
    body('notes').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { date } = req.params
    const { studyTime, questsCompleted, lessonsCompleted, focusScore, notes } =
      req.body

    const sessionDate = moment(date).format('YYYY-MM-DD')

    // Check if session exists
    const existingSession = await db.get(
      `
        SELECT * FROM daily_sessions WHERE date = ?
    `,
      [sessionDate]
    )

    if (!existingSession) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'SESSION_NOT_FOUND',
          message: 'Learning session not found for the specified date.'
        }
      })
    }

    // Build update query
    const updates = []
    const params = []

    if (studyTime !== undefined) {
      updates.push('study_time_minutes = ?')
      params.push(studyTime)
    }

    if (questsCompleted !== undefined) {
      updates.push('quests_completed = ?')
      params.push(questsCompleted)
    }

    if (lessonsCompleted !== undefined) {
      updates.push('lessons_completed = ?')
      params.push(lessonsCompleted)
    }

    if (focusScore !== undefined) {
      updates.push('focus_score = ?')
      params.push(focusScore)
    }

    if (notes !== undefined) {
      updates.push('notes = ?')
      params.push(notes)
    }

    updates.push('updated_at = datetime("now")')
    params.push(sessionDate)

    const updateQuery = `
        UPDATE daily_sessions 
        SET ${updates.join(', ')}
        WHERE date = ?
    `

    await db.run(updateQuery, params)

    // Get updated session
    const updatedSession = await db.get(
      `
        SELECT * FROM daily_sessions WHERE date = ?
    `,
      [sessionDate]
    )

    res.json({
      success: true,
      message: 'Learning session updated successfully',
      data: {
        date: updatedSession.date,
        studyTime: updatedSession.study_time_minutes,
        questsCompleted: updatedSession.quests_completed,
        lessonsCompleted: updatedSession.lessons_completed,
        focusScore: updatedSession.focus_score,
        notes: updatedSession.notes,
        updatedAt: updatedSession.updated_at
      }
    })
  })
)

// GET /api/v1/learning/spaced-repetition - Get items due for review
router.get(
  '/spaced-repetition',
  [
    query('dueDate')
      .optional()
      .isISO8601()
      .withMessage('Due date must be valid ISO8601 date'),
    query('limit').optional().isInt({ min: 1, max: 100 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { dueDate = moment().format('YYYY-MM-DD'), limit = 20 } = req.query

    // Get items due for review
    const dueItems = await db.all(
      `
        SELECT 
            sr.lesson_id, sr.item_id, sr.item_type, sr.ease_factor,
            sr.interval_days, sr.next_review_date, sr.review_count,
            sr.last_reviewed, sr.created_at,
            lp.lesson_title, lp.lesson_type, lp.phase, lp.week
        FROM spaced_repetition sr
        LEFT JOIN learning_progress lp ON sr.lesson_id = lp.lesson_id
        WHERE sr.next_review_date <= ?
        ORDER BY sr.next_review_date, sr.ease_factor
        LIMIT ?
    `,
      [dueDate, limit]
    )

    // Get upcoming items (next 7 days)
    const upcomingItems = await db.all(
      `
        SELECT 
            sr.lesson_id, sr.item_id, sr.item_type, sr.next_review_date,
            sr.interval_days, sr.review_count,
            lp.lesson_title, lp.phase, lp.week
        FROM spaced_repetition sr
        LEFT JOIN learning_progress lp ON sr.lesson_id = lp.lesson_id
        WHERE sr.next_review_date > ? AND sr.next_review_date <= date(?, '+7 days')
        ORDER BY sr.next_review_date
        LIMIT 10
    `,
      [dueDate, dueDate]
    )

    // Get review statistics
    const stats = await db.get(
      `
        SELECT 
            COUNT(*) as total_items,
            COUNT(CASE WHEN next_review_date <= ? THEN 1 END) as due_items,
            COUNT(CASE WHEN next_review_date > ? AND next_review_date <= date(?, '+7 days') THEN 1 END) as upcoming_items,
            AVG(ease_factor) as avg_ease_factor,
            AVG(interval_days) as avg_interval_days
        FROM spaced_repetition
    `,
      [dueDate, dueDate, dueDate]
    )

    res.json({
      success: true,
      data: {
        dueItems: dueItems.map(item => ({
          lessonId: item.lesson_id,
          itemId: item.item_id,
          itemType: item.item_type,
          lessonTitle: item.lesson_title,
          lessonType: item.lesson_type,
          phase: item.phase,
          week: item.week,
          easeFactor: item.ease_factor,
          intervalDays: item.interval_days,
          reviewCount: item.review_count,
          nextReviewDate: item.next_review_date,
          lastReviewed: item.last_reviewed
        })),
        upcomingItems: upcomingItems.map(item => ({
          lessonId: item.lesson_id,
          itemId: item.item_id,
          itemType: item.item_type,
          lessonTitle: item.lesson_title,
          phase: item.phase,
          week: item.week,
          nextReviewDate: item.next_review_date,
          intervalDays: item.interval_days,
          reviewCount: item.review_count
        })),
        statistics: {
          totalItems: stats.total_items || 0,
          dueItems: stats.due_items || 0,
          upcomingItems: stats.upcoming_items || 0,
          averageEaseFactor: parseFloat(
            (stats.avg_ease_factor || 2.5).toFixed(2)
          ),
          averageIntervalDays: Math.round(stats.avg_interval_days || 1)
        }
      }
    })
  })
)

// POST /api/v1/learning/spaced-repetition/:itemId/review - Record spaced repetition review
router.post(
  '/spaced-repetition/:itemId/review',
  [
    param('itemId').notEmpty().withMessage('Item ID is required'),
    body('quality')
      .isInt({ min: 0, max: 5 })
      .withMessage('Quality rating (0-5) is required'),
    body('timeSpent').optional().isInt({ min: 1 }),
    body('difficulty').optional().isInt({ min: 1, max: 5 }),
    body('notes').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { itemId } = req.params
    const { quality, timeSpent = 0, difficulty, notes } = req.body

    // Get existing spaced repetition item
    const item = await db.get(
      `
        SELECT * FROM spaced_repetition WHERE item_id = ?
    `,
      [itemId]
    )

    if (!item) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'SPACED_REPETITION_ITEM_NOT_FOUND',
          message: 'Spaced repetition item not found.'
        }
      })
    }

    // Calculate new ease factor and interval using SM-2 algorithm
    let newEaseFactor = item.ease_factor
    let newInterval = item.interval_days

    if (quality >= 3) {
      // Correct response
      if (item.review_count === 0) {
        newInterval = 1
      } else if (item.review_count === 1) {
        newInterval = 6
      } else {
        newInterval = Math.round(item.interval_days * newEaseFactor)
      }

      newEaseFactor =
        newEaseFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    } else {
      // Incorrect response
      newInterval = 1
    }

    // Ensure minimum ease factor
    if (newEaseFactor < 1.3) {
      newEaseFactor = 1.3
    }

    // Calculate next review date
    const nextReviewDate = moment()
      .add(newInterval, 'days')
      .format('YYYY-MM-DD')

    // Update spaced repetition item
    await db.run(
      `
        UPDATE spaced_repetition
        SET ease_factor = ?,
            interval_days = ?,
            next_review_date = ?,
            review_count = review_count + 1,
            last_reviewed = datetime('now'),
            updated_at = datetime('now')
        WHERE item_id = ?
    `,
      [newEaseFactor, newInterval, nextReviewDate, itemId]
    )

    // Record the review session
    await db.run(
      `
        INSERT INTO spaced_repetition_reviews (
            item_id, quality_rating, time_spent_minutes, difficulty_rating,
            notes, ease_factor_before, ease_factor_after, interval_before,
            interval_after, reviewed_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(item_id, reviewed_at) DO NOTHING
    `,
      [
        itemId,
        quality,
        timeSpent,
        difficulty || null,
        notes || null,
        item.ease_factor,
        newEaseFactor,
        item.interval_days,
        newInterval
      ]
    )

    res.json({
      success: true,
      message: 'Spaced repetition review recorded successfully!',
      data: {
        itemId,
        quality,
        newEaseFactor: parseFloat(newEaseFactor.toFixed(2)),
        newInterval,
        nextReviewDate,
        reviewCount: item.review_count + 1
      }
    })
  })
)

// GET /api/v1/learning/knowledge-graph - Get knowledge connections
router.get(
  '/knowledge-graph',
  [
    query('lessonId').optional().notEmpty(),
    query('conceptType')
      .optional()
      .isIn(['prerequisite', 'builds_on', 'relates_to', 'applies_in']),
    query('depth').optional().isInt({ min: 1, max: 3 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const { lessonId, conceptType, depth = 2 } = req.query

    let whereClause = []
    let params = []

    if (lessonId) {
      whereClause.push('(source_lesson_id = ? OR target_lesson_id = ?)')
      params.push(lessonId, lessonId)
    }

    if (conceptType) {
      whereClause.push('connection_type = ?')
      params.push(conceptType)
    }

    const whereSQL =
      whereClause.length > 0 ? `WHERE ${whereClause.join(' AND ')}` : ''

    // Get knowledge connections
    const connections = await db.all(
      `
        SELECT 
            kg.source_lesson_id, kg.target_lesson_id, kg.connection_type,
            kg.strength, kg.description, kg.created_at,
            lp1.lesson_title as source_title, lp1.phase as source_phase, lp1.week as source_week,
            lp2.lesson_title as target_title, lp2.phase as target_phase, lp2.week as target_week
        FROM knowledge_graph kg
        LEFT JOIN learning_progress lp1 ON kg.source_lesson_id = lp1.lesson_id
        LEFT JOIN learning_progress lp2 ON kg.target_lesson_id = lp2.lesson_id
        ${whereSQL}
        ORDER BY kg.strength DESC, kg.created_at DESC
    `,
      params
    )

    // Build nodes and edges for graph visualization
    const nodes = new Map()
    const edges = []

    connections.forEach(conn => {
      // Add source node
      if (!nodes.has(conn.source_lesson_id)) {
        nodes.set(conn.source_lesson_id, {
          id: conn.source_lesson_id,
          title: conn.source_title,
          phase: conn.source_phase,
          week: conn.source_week,
          type: 'lesson'
        })
      }

      // Add target node
      if (!nodes.has(conn.target_lesson_id)) {
        nodes.set(conn.target_lesson_id, {
          id: conn.target_lesson_id,
          title: conn.target_title,
          phase: conn.target_phase,
          week: conn.target_week,
          type: 'lesson'
        })
      }

      // Add edge
      edges.push({
        source: conn.source_lesson_id,
        target: conn.target_lesson_id,
        type: conn.connection_type,
        strength: conn.strength,
        description: conn.description
      })
    })

    res.json({
      success: true,
      data: {
        connections: connections.map(conn => ({
          source: {
            id: conn.source_lesson_id,
            title: conn.source_title,
            phase: conn.source_phase,
            week: conn.source_week
          },
          target: {
            id: conn.target_lesson_id,
            title: conn.target_title,
            phase: conn.target_phase,
            week: conn.target_week
          },
          type: conn.connection_type,
          strength: conn.strength,
          description: conn.description,
          createdAt: conn.created_at
        })),
        graph: {
          nodes: Array.from(nodes.values()),
          edges
        }
      }
    })
  })
)

// POST /api/v1/learning/knowledge-graph - Add knowledge connection
router.post(
  '/knowledge-graph',
  [
    body('sourceLessonId')
      .notEmpty()
      .withMessage('Source lesson ID is required'),
    body('targetLessonId')
      .notEmpty()
      .withMessage('Target lesson ID is required'),
    body('connectionType').isIn([
      'prerequisite',
      'builds_on',
      'relates_to',
      'applies_in'
    ]),
    body('strength')
      .isFloat({ min: 0.1, max: 1.0 })
      .withMessage('Strength must be between 0.1 and 1.0'),
    body('description').optional().isString().isLength({ max: 500 })
  ],
  handleValidationErrors,
  asyncErrorHandler(async (req, res) => {
    const {
      sourceLessonId,
      targetLessonId,
      connectionType,
      strength,
      description
    } = req.body

    // Check if both lessons exist
    const sourceLesson = await db.get(
      `
        SELECT lesson_id FROM learning_progress WHERE lesson_id = ?
    `,
      [sourceLessonId]
    )

    const targetLesson = await db.get(
      `
        SELECT lesson_id FROM learning_progress WHERE lesson_id = ?
    `,
      [targetLessonId]
    )

    if (!sourceLesson || !targetLesson) {
      return res.status(400).json({
        success: false,
        error: {
          type: 'INVALID_LESSON_REFERENCE',
          message: 'One or both lesson IDs do not exist.'
        }
      })
    }

    // Insert knowledge connection
    const result = await db.run(
      `
        INSERT OR REPLACE INTO knowledge_graph (
            source_lesson_id, target_lesson_id, connection_type,
            strength, description, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    `,
      [
        sourceLessonId,
        targetLessonId,
        connectionType,
        strength,
        description || null
      ]
    )

    res.status(201).json({
      success: true,
      message: 'Knowledge connection created successfully!',
      data: {
        id: result.lastID,
        sourceLessonId,
        targetLessonId,
        connectionType,
        strength,
        description
      }
    })
  })
)

// GET /api/v1/learning/next-lesson - Get next lesson recommendation
router.get(
  '/next-lesson',
  asyncErrorHandler(async (req, res) => {
    // Get user's current progress
    const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')

    if (!profile) {
      return res.status(404).json({
        success: false,
        error: {
          type: 'USER_PROFILE_NOT_FOUND',
          message: 'User profile not found. Please complete setup first.'
        }
      })
    }

    // Find next lesson based on current progress
    const nextLesson = await db.get(
      `
        SELECT 
            lesson_id, lesson_title, lesson_type, phase, week,
            status, understanding_score, confidence_level
        FROM learning_progress
        WHERE phase = ? AND week = ? AND status = 'not_started'
        ORDER BY lesson_id
        LIMIT 1
    `,
      [profile.current_phase, profile.current_week]
    )

    // If no lessons in current week, look for next week
    let recommendation = null

    if (nextLesson) {
      recommendation = {
        type: 'continue_current_week',
        lesson: {
          id: nextLesson.lesson_id,
          title: nextLesson.lesson_title,
          type: nextLesson.lesson_type,
          phase: nextLesson.phase,
          week: nextLesson.week
        },
        reason: `Continue with Week ${nextLesson.week} lessons in Phase ${nextLesson.phase}`
      }
    } else {
      // Look for next available lesson in current phase
      const nextPhaseLesson = await db.get(
        `
            SELECT 
                lesson_id, lesson_title, lesson_type, phase, week,
                status, understanding_score, confidence_level
            FROM learning_progress
            WHERE phase = ? AND week > ? AND status = 'not_started'
            ORDER BY week, lesson_id
            LIMIT 1
        `,
        [profile.current_phase, profile.current_week]
      )

      if (nextPhaseLesson) {
        recommendation = {
          type: 'advance_to_next_week',
          lesson: {
            id: nextPhaseLesson.lesson_id,
            title: nextPhaseLesson.lesson_title,
            type: nextPhaseLesson.lesson_type,
            phase: nextPhaseLesson.phase,
            week: nextPhaseLesson.week
          },
          reason: `Advance to Week ${nextPhaseLesson.week} in Phase ${nextPhaseLesson.phase}`
        }
      } else {
        // Look for next phase
        const nextPhase = await db.get(
          `
                SELECT 
                    lesson_id, lesson_title, lesson_type, phase, week,
                    status, understanding_score, confidence_level
                FROM learning_progress
                WHERE phase > ? AND status = 'not_started'
                ORDER BY phase, week, lesson_id
                LIMIT 1
            `,
          [profile.current_phase]
        )

        if (nextPhase) {
          recommendation = {
            type: 'advance_to_next_phase',
            lesson: {
              id: nextPhase.lesson_id,
              title: nextPhase.lesson_title,
              type: nextPhase.lesson_type,
              phase: nextPhase.phase,
              week: nextPhase.week
            },
            reason: `Advance to Phase ${nextPhase.phase} - ${nextPhase.lesson_title}`
          }
        } else {
          recommendation = {
            type: 'curriculum_complete',
            lesson: null,
            reason:
              'Congratulations! You have completed the entire Neural Odyssey curriculum!'
          }
        }
      }
    }

    // Get recent activity context
    const recentSessions = await db.all(`
        SELECT date, study_time_minutes, focus_score
        FROM daily_sessions
        ORDER BY date DESC
        LIMIT 7
    `)

    // Get spaced repetition items due
    const dueReviews = await db.all(`
        SELECT COUNT(*) as count
        FROM spaced_repetition
        WHERE next_review_date <= date('now')
    `)

    res.json({
      success: true,
      data: {
        recommendation,
        context: {
          currentPhase: profile.current_phase,
          currentWeek: profile.current_week,
          totalStudyTime: profile.total_study_time_minutes,
          currentStreak: profile.streak_days,
          dueReviews: dueReviews[0]?.count || 0,
          recentActivity: recentSessions.map(session => ({
            date: session.date,
            studyTime: session.study_time_minutes,
            focusScore: session.focus_score
          }))
        }
      }
    })
  })
)

// Export the router
module.exports = router
