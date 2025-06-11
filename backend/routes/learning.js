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

const express = require('express');
const { body, param, query, validationResult } = require('express-validator');
const router = express.Router();
const db = require('../config/db');
const moment = require('moment');

// Middleware to handle validation errors
const handleValidationErrors = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            error: 'Validation failed',
            details: errors.array()
        });
    }
    next();
};

// GET /api/v1/learning/progress - Get overall learning progress
router.get('/progress', async (req, res, next) => {
    try {
        // Get user profile and current progress
        const profile = await db.get('SELECT * FROM user_profile WHERE id = 1');
        
        // Get overall progress statistics
        const overallStats = await db.get(`
            SELECT 
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_lessons,
                SUM(time_spent_minutes) as total_study_minutes,
                AVG(CASE WHEN mastery_score IS NOT NULL THEN mastery_score END) as avg_mastery_score,
                MAX(updated_at) as last_activity
            FROM learning_progress
        `);

        // Get progress by phase
        const phaseProgress = await db.query(`
            SELECT 
                phase,
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                SUM(time_spent_minutes) as time_spent_minutes,
                AVG(CASE WHEN mastery_score IS NOT NULL THEN mastery_score END) as avg_mastery_score
            FROM learning_progress
            GROUP BY phase
            ORDER BY phase
        `);

        // Get current week's progress
        const currentWeekProgress = await db.query(`
            SELECT *
            FROM learning_progress
            WHERE phase = ? AND week = ?
            ORDER BY lesson_id
        `, [profile.current_phase, profile.current_week]);

        // Calculate completion percentage
        const completionPercentage = overallStats.total_lessons > 0 
            ? Math.round((overallStats.completed_lessons / overallStats.total_lessons) * 100)
            : 0;

        // Get recent activity (last 7 days)
        const recentSessions = await db.query(`
            SELECT session_date, session_type, duration_minutes, focus_score
            FROM daily_sessions
            WHERE session_date >= date('now', '-7 days')
            ORDER BY session_date DESC
        `);

        res.json({
            profile: {
                username: profile.username,
                current_phase: profile.current_phase,
                current_week: profile.current_week,
                current_streak_days: profile.current_streak_days,
                longest_streak_days: profile.longest_streak_days,
                total_study_minutes: profile.total_study_minutes
            },
            overall: {
                ...overallStats,
                completion_percentage: completionPercentage
            },
            phases: phaseProgress,
            current_week: currentWeekProgress,
            recent_activity: recentSessions
        });

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/progress/phase/:phase - Get phase-specific progress
router.get('/progress/phase/:phase', [
    param('phase').isInt({ min: 1, max: 4 }).withMessage('Phase must be between 1 and 4')
], handleValidationErrors, async (req, res, next) => {
    try {
        const { phase } = req.params;

        // Get phase overview
        const phaseStats = await db.get(`
            SELECT 
                phase,
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                SUM(time_spent_minutes) as total_time_minutes,
                AVG(CASE WHEN mastery_score IS NOT NULL THEN mastery_score END) as avg_mastery_score
            FROM learning_progress
            WHERE phase = ?
        `, [phase]);

        // Get progress by week within the phase
        const weeklyProgress = await db.query(`
            SELECT 
                week,
                COUNT(*) as total_lessons,
                COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                SUM(time_spent_minutes) as time_spent_minutes
            FROM learning_progress
            WHERE phase = ?
            GROUP BY week
            ORDER BY week
        `, [phase]);

        // Get all lessons in the phase
        const lessons = await db.query(`
            SELECT *
            FROM learning_progress
            WHERE phase = ?
            ORDER BY week, lesson_id
        `, [phase]);

        const phaseNames = {
            1: 'Mathematical Foundations and Historical Context',
            2: 'Core Machine Learning with Deep Understanding',
            3: 'Advanced Topics and Modern AI',
            4: 'Mastery and Innovation'
        };

        res.json({
            phase: parseInt(phase),
            title: phaseNames[phase] || `Phase ${phase}`,
            stats: phaseStats,
            weekly_progress: weeklyProgress,
            lessons: lessons
        });

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/lessons - Get lessons with filtering
router.get('/lessons', [
    query('phase').optional().isInt({ min: 1, max: 4 }),
    query('week').optional().isInt({ min: 1, max: 48 }),
    query('status').optional().isIn(['not_started', 'in_progress', 'completed', 'mastered']),
    query('type').optional().isIn(['theory', 'math', 'visual', 'coding'])
], handleValidationErrors, async (req, res, next) => {
    try {
        const { phase, week, status, type } = req.query;
        
        let sql = 'SELECT * FROM learning_progress WHERE 1=1';
        const params = [];

        if (phase) {
            sql += ' AND phase = ?';
            params.push(phase);
        }

        if (week) {
            sql += ' AND week = ?';
            params.push(week);
        }

        if (status) {
            sql += ' AND status = ?';
            params.push(status);
        }

        if (type) {
            sql += ' AND lesson_type = ?';
            params.push(type);
        }

        sql += ' ORDER BY phase, week, lesson_id';

        const lessons = await db.query(sql, params);

        res.json({
            lessons,
            total: lessons.length,
            filters: { phase, week, status, type }
        });

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/lessons/:id - Get specific lesson details
router.get('/lessons/:id', [
    param('id').notEmpty().withMessage('Lesson ID is required')
], handleValidationErrors, async (req, res, next) => {
    try {
        const { id } = req.params;

        const lesson = await db.get(`
            SELECT * FROM learning_progress 
            WHERE lesson_id = ?
        `, [id]);

        if (!lesson) {
            return res.status(404).json({
                error: 'Lesson not found',
                lesson_id: id
            });
        }

        // Get related quests for this lesson
        const relatedQuests = await db.query(`
            SELECT * FROM quest_completions
            WHERE related_lesson_id = ?
            ORDER BY completed_at DESC
        `, [id]);

        // Get spaced repetition items related to this lesson
        const spacedRepetitionItems = await db.query(`
            SELECT * FROM spaced_repetition
            WHERE concept_id LIKE ?
            ORDER BY next_review_date
        `, [`%${id}%`]);

        res.json({
            lesson,
            related_quests: relatedQuests,
            spaced_repetition: spacedRepetitionItems
        });

    } catch (error) {
        next(error);
    }
});

// PUT /api/v1/learning/lessons/:id/progress - Update lesson progress
router.put('/lessons/:id/progress', [
    param('id').notEmpty().withMessage('Lesson ID is required'),
    body('status').optional().isIn(['not_started', 'in_progress', 'completed', 'mastered']),
    body('completion_percentage').optional().isInt({ min: 0, max: 100 }),
    body('time_spent_minutes').optional().isInt({ min: 0 }),
    body('mastery_score').optional().isFloat({ min: 0, max: 1 }),
    body('notes').optional().isString()
], handleValidationErrors, async (req, res, next) => {
    try {
        const { id } = req.params;
        const { status, completion_percentage, time_spent_minutes, mastery_score, notes } = req.body;

        // Check if lesson exists
        const existingLesson = await db.get('SELECT * FROM learning_progress WHERE lesson_id = ?', [id]);
        if (!existingLesson) {
            return res.status(404).json({
                error: 'Lesson not found',
                lesson_id: id
            });
        }

        // Build update query dynamically
        const updates = [];
        const params = [];

        if (status !== undefined) {
            updates.push('status = ?');
            params.push(status);
        }

        if (completion_percentage !== undefined) {
            updates.push('completion_percentage = ?');
            params.push(completion_percentage);
        }

        if (time_spent_minutes !== undefined) {
            updates.push('time_spent_minutes = ?');
            params.push(time_spent_minutes);
        }

        if (mastery_score !== undefined) {
            updates.push('mastery_score = ?');
            params.push(mastery_score);
        }

        if (notes !== undefined) {
            updates.push('notes = ?');
            params.push(notes);
        }

        // Always update timestamp
        updates.push('updated_at = CURRENT_TIMESTAMP');

        // If marking as completed, set completed_at
        if (status === 'completed' || status === 'mastered') {
            updates.push('completed_at = CURRENT_TIMESTAMP');
        }

        params.push(id);

        const sql = `UPDATE learning_progress SET ${updates.join(', ')} WHERE lesson_id = ?`;
        const result = await db.run(sql, params);

        // Get updated lesson
        const updatedLesson = await db.get('SELECT * FROM learning_progress WHERE lesson_id = ?', [id]);

        // Award skill points if lesson was completed
        if (status === 'completed' || status === 'mastered') {
            const pointsEarned = status === 'mastered' ? 15 : 10;
            await db.run(`
                INSERT INTO skill_points (category, points_earned, reason, related_lesson_id)
                VALUES (?, ?, ?, ?)
            `, ['theory', pointsEarned, `Completed lesson: ${updatedLesson.lesson_title}`, id]);
        }

        res.json({
            message: 'Lesson progress updated successfully',
            lesson: updatedLesson,
            changes: result.changes
        });

    } catch (error) {
        next(error);
    }
});

// POST /api/v1/learning/lessons/:id/complete - Mark lesson as completed (convenience endpoint)
router.post('/lessons/:id/complete', [
    param('id').notEmpty().withMessage('Lesson ID is required'),
    body('time_spent_minutes').optional().isInt({ min: 0 }),
    body('mastery_score').optional().isFloat({ min: 0, max: 1 }),
    body('notes').optional().isString()
], handleValidationErrors, async (req, res, next) => {
    try {
        const { id } = req.params;
        const { time_spent_minutes, mastery_score, notes } = req.body;

        const updates = {
            status: mastery_score && mastery_score >= 0.8 ? 'mastered' : 'completed',
            completion_percentage: 100,
            completed_at: new Date().toISOString()
        };

        if (time_spent_minutes !== undefined) updates.time_spent_minutes = time_spent_minutes;
        if (mastery_score !== undefined) updates.mastery_score = mastery_score;
        if (notes !== undefined) updates.notes = notes;

        // Use the progress update endpoint logic
        req.body = updates;
        req.params = { id };
        
        // Call the progress update handler
        return router.handle({ method: 'PUT', url: `/lessons/${id}/progress`, body: updates, params: { id } }, res, next);

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/quests - Get quest completions
router.get('/quests', [
    query('phase').optional().isInt({ min: 1, max: 4 }),
    query('status').optional().isIn(['attempted', 'completed', 'mastered']),
    query('type').optional().isIn(['coding_exercise', 'implementation_project', 'theory_quiz', 'practical_application'])
], handleValidationErrors, async (req, res, next) => {
    try {
        const { phase, status, type } = req.query;
        
        let sql = 'SELECT * FROM quest_completions WHERE 1=1';
        const params = [];

        if (phase) {
            sql += ' AND phase = ?';
            params.push(phase);
        }

        if (status) {
            sql += ' AND status = ?';
            params.push(status);
        }

        if (type) {
            sql += ' AND quest_type = ?';
            params.push(type);
        }

        sql += ' ORDER BY phase, week, completed_at DESC';

        const quests = await db.query(sql, params);

        // Get quest statistics
        const stats = await db.get(`
            SELECT 
                COUNT(*) as total_quests,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_quests,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_quests,
                AVG(time_to_complete_minutes) as avg_completion_time,
                AVG(attempts_count) as avg_attempts
            FROM quest_completions
            ${phase ? 'WHERE phase = ?' : ''}
        `, phase ? [phase] : []);

        res.json({
            quests,
            stats,
            total: quests.length,
            filters: { phase, status, type }
        });

    } catch (error) {
        next(error);
    }
});

// POST /api/v1/learning/quests - Create new quest completion
router.post('/quests', [
    body('quest_id').notEmpty().withMessage('Quest ID is required'),
    body('quest_title').notEmpty().withMessage('Quest title is required'),
    body('quest_type').isIn(['coding_exercise', 'implementation_project', 'theory_quiz', 'practical_application']),
    body('phase').isInt({ min: 1, max: 4 }),
    body('week').isInt({ min: 1, max: 48 }),
    body('difficulty_level').isInt({ min: 1, max: 5 }),
    body('status').optional().isIn(['attempted', 'completed', 'mastered']),
    body('code_solution').optional().isString(),
    body('execution_result').optional().isString(),
    body('time_to_complete_minutes').optional().isInt({ min: 0 }),
    body('self_reflection').optional().isString()
], handleValidationErrors, async (req, res, next) => {
    try {
        const {
            quest_id, quest_title, quest_type, phase, week, difficulty_level,
            status = 'attempted', code_solution, execution_result,
            time_to_complete_minutes, self_reflection
        } = req.body;

        const result = await db.run(`
            INSERT INTO quest_completions (
                quest_id, quest_title, quest_type, phase, week, difficulty_level,
                status, code_solution, execution_result, time_to_complete_minutes, self_reflection
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `, [
            quest_id, quest_title, quest_type, phase, week, difficulty_level,
            status, code_solution, execution_result, time_to_complete_minutes, self_reflection
        ]);

        // Award skill points
        const basePoints = difficulty_level * 5;
        const statusMultiplier = { attempted: 0.5, completed: 1, mastered: 1.5 };
        const pointsEarned = Math.round(basePoints * statusMultiplier[status]);

        await db.run(`
            INSERT INTO skill_points (category, points_earned, reason, related_quest_id)
            VALUES (?, ?, ?, ?)
        `, ['programming', pointsEarned, `${status} quest: ${quest_title}`, quest_id]);

        // Get the created quest
        const createdQuest = await db.get('SELECT * FROM quest_completions WHERE id = ?', [result.lastID]);

        res.status(201).json({
            message: 'Quest completion recorded successfully',
            quest: createdQuest,
            points_earned: pointsEarned
        });

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/sessions - Get daily learning sessions
router.get('/sessions', [
    query('days').optional().isInt({ min: 1, max: 365 }).withMessage('Days must be between 1 and 365'),
    query('type').optional().isIn(['math', 'coding', 'visual_projects', 'real_applications'])
], handleValidationErrors, async (req, res, next) => {
    try {
        const { days = 30, type } = req.query;
        
        let sql = `
            SELECT * FROM daily_sessions 
            WHERE session_date >= date('now', '-${days} days')
        `;
        const params = [];

        if (type) {
            sql += ' AND session_type = ?';
            params.push(type);
        }

        sql += ' ORDER BY session_date DESC, start_time DESC';

        const sessions = await db.query(sql, params);

        // Get session statistics
        const stats = await db.get(`
            SELECT 
                COUNT(*) as total_sessions,
                SUM(duration_minutes) as total_minutes,
                AVG(duration_minutes) as avg_duration,
                AVG(focus_score) as avg_focus,
                AVG(energy_level) as avg_energy,
                MAX(session_date) as last_session_date
            FROM daily_sessions
            WHERE session_date >= date('now', '-${days} days')
            ${type ? 'AND session_type = ?' : ''}
        `, type ? [type] : []);

        res.json({
            sessions,
            stats,
            period_days: days,
            filter: { type }
        });

    } catch (error) {
        next(error);
    }
});

// POST /api/v1/learning/sessions - Start new learning session
router.post('/sessions', [
    body('session_type').isIn(['math', 'coding', 'visual_projects', 'real_applications']),
    body('planned_duration_minutes').optional().isInt({ min: 1, max: 480 })
], handleValidationErrors, async (req, res, next) => {
    try {
        const { session_type, planned_duration_minutes = 25 } = req.body;

        const result = await db.run(`
            INSERT INTO daily_sessions (session_date, session_type, start_time)
            VALUES (date('now'), ?, datetime('now'))
        `, [session_type]);

        const session = await db.get('SELECT * FROM daily_sessions WHERE id = ?', [result.lastID]);

        res.status(201).json({
            message: 'Learning session started',
            session,
            planned_duration: planned_duration_minutes
        });

    } catch (error) {
        next(error);
    }
});

// PUT /api/v1/learning/sessions/:id - Update/end learning session
router.put('/sessions/:id', [
    param('id').isInt().withMessage('Session ID must be an integer'),
    body('end_time').optional().isISO8601().withMessage('End time must be valid ISO date'),
    body('focus_score').optional().isInt({ min: 1, max: 10 }),
    body('energy_level').optional().isInt({ min: 1, max: 10 }),
    body('completed_goals').optional().isString(),
    body('session_notes').optional().isString()
], handleValidationErrors, async (req, res, next) => {
    try {
        const { id } = req.params;
        const { end_time, focus_score, energy_level, completed_goals, session_notes } = req.body;

        // Check if session exists
        const existingSession = await db.get('SELECT * FROM daily_sessions WHERE id = ?', [id]);
        if (!existingSession) {
            return res.status(404).json({
                error: 'Session not found',
                session_id: id
            });
        }

        // Build update query
        const updates = [];
        const params = [];

        if (end_time !== undefined) {
            updates.push('end_time = ?');
            params.push(end_time);
            
            // Calculate duration if end_time is provided
            const startTime = new Date(existingSession.start_time);
            const endDateTime = new Date(end_time);
            const durationMinutes = Math.round((endDateTime - startTime) / (1000 * 60));
            
            updates.push('duration_minutes = ?');
            params.push(durationMinutes);
        }

        if (focus_score !== undefined) {
            updates.push('focus_score = ?');
            params.push(focus_score);
        }

        if (energy_level !== undefined) {
            updates.push('energy_level = ?');
            params.push(energy_level);
        }

        if (completed_goals !== undefined) {
            updates.push('completed_goals = ?');
            params.push(completed_goals);
        }

        if (session_notes !== undefined) {
            updates.push('session_notes = ?');
            params.push(session_notes);
        }

        params.push(id);

        const sql = `UPDATE daily_sessions SET ${updates.join(', ')} WHERE id = ?`;
        await db.run(sql, params);

        // Get updated session
        const updatedSession = await db.get('SELECT * FROM daily_sessions WHERE id = ?', [id]);

        res.json({
            message: 'Session updated successfully',
            session: updatedSession
        });

    } catch (error) {
        next(error);
    }
});

// GET /api/v1/learning/next-lesson - Get next recommended lesson
router.get('/next-lesson', async (req, res, next) => {
    try {
        // Get current user phase and week
        const profile = await db.get('SELECT current_phase, current_week FROM user_profile WHERE id = 1');

        // Find next incomplete lesson in current week
        let nextLesson = await db.get(`
            SELECT * FROM learning_progress
            WHERE phase = ? AND week = ? AND status = 'not_started'
            ORDER BY lesson_id
            LIMIT 1
        `, [profile.current_phase, profile.current_week]);

        // If no lessons in current week, find next in-progress lesson
        if (!nextLesson) {
            nextLesson = await db.get(`
                SELECT * FROM learning_progress
                WHERE phase = ? AND week = ? AND status = 'in_progress'
                ORDER BY lesson_id
                LIMIT 1
            `, [profile.current_phase, profile.current_week]);
        }

        // If current week is complete, find first lesson of next week
        if (!nextLesson) {
            nextLesson = await db.get(`
                SELECT * FROM learning_progress
                WHERE phase = ? AND week = ? AND status = 'not_started'
                ORDER BY lesson_id
                LIMIT 1
            `, [profile.current_phase, profile.current_week + 1]);
        }

        // If current phase is complete, find first lesson of next phase
        if (!nextLesson) {
            nextLesson = await db.get(`
                SELECT * FROM learning_progress
                WHERE phase = ? AND week = 1 AND status = 'not_started'
                ORDER BY lesson_id
                LIMIT 1
            `, [profile.current_phase + 1]);
        }

        if (!nextLesson) {
            return res.json({
                message: 'Congratulations! You have completed your Neural Odyssey journey!',
                completed: true
            });
        }

        // Get context: previous and upcoming lessons
        const context = await db.query(`
            SELECT * FROM learning_progress
            WHERE phase = ? AND week = ?
            ORDER BY lesson_id
        `, [nextLesson.phase, nextLesson.week]);

        res.json({
            next_lesson: nextLesson,
            current_week_context: context,
            recommendation: {
                type: 'daily_rotation',
                suggested_session_type: nextLesson.lesson_type,
                estimated_duration: 25 // Pomodoro session
            }
        });

    } catch (error) {
        next(error);
    }
});

module.exports = router;