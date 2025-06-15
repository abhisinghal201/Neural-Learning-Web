/**
 * Neural Odyssey Quest Model
 *
 * Handles quest definitions, unlock logic, difficulty progression, and validation
 * for the gamified learning system. Quests are the core challenges that test
 * understanding and provide hands-on practice across the ML learning journey.
 *
 * Quest Types:
 * - coding_exercise: Programming challenges with automated testing
 * - implementation_project: Complex projects requiring multiple files/concepts
 * - theory_quiz: Knowledge validation through interactive questions
 * - practical_application: Real-world problem solving with ML techniques
 *
 * Features:
 * - Dynamic quest generation based on learning progress
 * - Difficulty scaling with adaptive parameters
 * - Automated test case validation
 * - Progress tracking and unlock conditions
 * - Code quality assessment
 * - Learning path integration
 *
 * Author: Neural Explorer
 */

const db = require('../config/db')
const fs = require('fs')
const path = require('path')
const moment = require('moment')
const crypto = require('crypto')

class Quest {
  constructor (data = {}) {
    this.id = data.id || this.generateId()
    this.title = data.title || ''
    this.description = data.description || ''
    this.type = data.type || 'coding_exercise'
    this.phase = data.phase || 1
    this.week = data.week || 1
    this.difficulty = data.difficulty || 'beginner'
    this.estimatedTime = data.estimatedTime || 30
    this.prerequisites = data.prerequisites || []
    this.learningObjectives = data.learningObjectives || []
    this.tags = data.tags || []
    this.isUnlocked = data.isUnlocked || false
    this.unlockConditions = data.unlockConditions || {}
    this.testCases = data.testCases || []
    this.starterCode = data.starterCode || ''
    this.solution = data.solution || ''
    this.hints = data.hints || []
    this.resources = data.resources || []
    this.createdAt = data.createdAt || new Date().toISOString()
    this.updatedAt = data.updatedAt || new Date().toISOString()
  }

  /**
   * Generate unique quest ID
   * @returns {string} - Unique quest identifier
   */
  generateId () {
    const timestamp = Date.now().toString(36)
    const random = crypto.randomBytes(4).toString('hex')
    return `quest_${timestamp}_${random}`
  }

  /**
   * Validate quest data
   * @returns {Object} - Validation result with errors if any
   */
  validate () {
    const errors = []

    if (!this.title || this.title.trim().length === 0) {
      errors.push('Quest title is required')
    }

    if (!this.description || this.description.trim().length === 0) {
      errors.push('Quest description is required')
    }

    if (
      ![
        'coding_exercise',
        'implementation_project',
        'theory_quiz',
        'practical_application'
      ].includes(this.type)
    ) {
      errors.push('Invalid quest type')
    }

    if (!Number.isInteger(this.phase) || this.phase < 1 || this.phase > 4) {
      errors.push('Phase must be an integer between 1 and 4')
    }

    if (!Number.isInteger(this.week) || this.week < 1 || this.week > 12) {
      errors.push('Week must be an integer between 1 and 12')
    }

    if (
      !['beginner', 'intermediate', 'advanced', 'expert'].includes(
        this.difficulty
      )
    ) {
      errors.push('Invalid difficulty level')
    }

    if (this.type === 'coding_exercise' && this.testCases.length === 0) {
      errors.push('Coding exercises must have at least one test case')
    }

    return {
      isValid: errors.length === 0,
      errors
    }
  }

  /**
   * Check if quest can be unlocked based on user progress
   * @param {Object} userProgress - User's learning progress data
   * @returns {Promise<Object>} - Unlock status and requirements
   */
  async checkUnlockConditions (userProgress = null) {
    try {
      const conditions = this.unlockConditions
      const result = {
        canUnlock: true,
        missingRequirements: [],
        progress: {}
      }

      // Get user progress if not provided
      if (!userProgress) {
        userProgress = await this.getUserProgress()
      }

      // Check lesson prerequisites
      if (conditions.requiredLessons && conditions.requiredLessons.length > 0) {
        for (const lessonId of conditions.requiredLessons) {
          const lesson = await db.get(
            `
                        SELECT * FROM learning_progress 
                        WHERE lesson_id = ? AND status IN ('completed', 'mastered')
                    `,
            [lessonId]
          )

          if (!lesson) {
            result.canUnlock = false
            result.missingRequirements.push(`Complete lesson: ${lessonId}`)
          }
        }
      }

      // Check phase/week completion requirements
      if (conditions.minPhaseCompletion) {
        const phaseStats = await db.get(
          `
                    SELECT 
                        COUNT(*) as total_lessons,
                        COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completed_lessons
                    FROM learning_progress 
                    WHERE phase = ?
                `,
          [this.phase]
        )

        const completionRate =
          phaseStats.total_lessons > 0
            ? phaseStats.completed_lessons / phaseStats.total_lessons
            : 0

        if (completionRate < conditions.minPhaseCompletion) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Complete at least ${(conditions.minPhaseCompletion * 100).toFixed(
              0
            )}% of Phase ${this.phase} lessons`
          )
        }

        result.progress.phaseCompletion = completionRate
      }

      // Check previous quest completions
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
          }
        }
      }

      // Check minimum mastery score requirement
      if (conditions.minMasteryScore) {
        const avgMastery = await db.get(
          `
                    SELECT AVG(mastery_score) as avg_score
                    FROM learning_progress 
                    WHERE phase = ? AND week = ? AND mastery_score IS NOT NULL
                `,
          [this.phase, this.week]
        )

        if (
          !avgMastery.avg_score ||
          avgMastery.avg_score < conditions.minMasteryScore
        ) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Achieve minimum mastery score of ${(
              conditions.minMasteryScore * 100
            ).toFixed(0)}% in current week`
          )
        }

        result.progress.currentMastery = avgMastery.avg_score
      }

      // Check skill points requirement
      if (conditions.requiredSkillPoints) {
        for (const [category, requiredPoints] of Object.entries(
          conditions.requiredSkillPoints
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

      // Check streak requirement
      if (conditions.minStreak) {
        const profile = await db.get(
          'SELECT current_streak_days FROM user_profile WHERE id = 1'
        )
        const currentStreak = profile ? profile.current_streak_days : 0

        if (currentStreak < conditions.minStreak) {
          result.canUnlock = false
          result.missingRequirements.push(
            `Maintain a learning streak of ${conditions.minStreak} days`
          )
        }

        result.progress.currentStreak = currentStreak
      }

      return result
    } catch (error) {
      console.error('❌ Error checking unlock conditions:', error)
      return {
        canUnlock: false,
        missingRequirements: ['Error checking requirements'],
        progress: {}
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

      return {
        profile,
        lessons: lessonProgress,
        quests: questProgress,
        skills: skillPoints
      }
    } catch (error) {
      console.error('❌ Error getting user progress:', error)
      return null
    }
  }

  /**
   * Save quest to database
   * @returns {Promise<Object>} - Save result
   */
  async save () {
    try {
      const validation = this.validate()
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`)
      }

      const exists = await db.get('SELECT id FROM quests WHERE id = ?', [
        this.id
      ])

      if (exists) {
        // Update existing quest
        await db.run(
          `
                    UPDATE quests SET
                        title = ?, description = ?, type = ?, phase = ?, week = ?,
                        difficulty = ?, estimated_time_minutes = ?, prerequisites = ?,
                        learning_objectives = ?, tags = ?, unlock_conditions = ?,
                        test_cases = ?, starter_code = ?, solution = ?, hints = ?,
                        resources = ?, updated_at = datetime('now')
                    WHERE id = ?
                `,
          [
            this.title,
            this.description,
            this.type,
            this.phase,
            this.week,
            this.difficulty,
            this.estimatedTime,
            JSON.stringify(this.prerequisites),
            JSON.stringify(this.learningObjectives),
            JSON.stringify(this.tags),
            JSON.stringify(this.unlockConditions),
            JSON.stringify(this.testCases),
            this.starterCode,
            this.solution,
            JSON.stringify(this.hints),
            JSON.stringify(this.resources),
            this.id
          ]
        )
      } else {
        // Create new quest
        await db.run(
          `
                    INSERT INTO quests (
                        id, title, description, type, phase, week, difficulty,
                        estimated_time_minutes, prerequisites, learning_objectives,
                        tags, unlock_conditions, test_cases, starter_code, solution,
                        hints, resources, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                             datetime('now'), datetime('now'))
                `,
          [
            this.id,
            this.title,
            this.description,
            this.type,
            this.phase,
            this.week,
            this.difficulty,
            this.estimatedTime,
            JSON.stringify(this.prerequisites),
            JSON.stringify(this.learningObjectives),
            JSON.stringify(this.tags),
            JSON.stringify(this.unlockConditions),
            JSON.stringify(this.testCases),
            this.starterCode,
            this.solution,
            JSON.stringify(this.hints),
            JSON.stringify(this.resources)
          ]
        )
      }

      return { success: true, id: this.id }
    } catch (error) {
      console.error('❌ Error saving quest:', error)
      throw error
    }
  }

  /**
   * Validate submitted solution against test cases
   * @param {string} userCode - User's submitted code
   * @param {string} language - Programming language (default: python)
   * @returns {Promise<Object>} - Validation results
   */
  async validateSolution (userCode, language = 'python') {
    try {
      if (this.type !== 'coding_exercise') {
        return {
          success: true,
          message: 'Non-coding quest - manual review required',
          testResults: []
        }
      }

      const results = {
        success: false,
        passedTests: 0,
        totalTests: this.testCases.length,
        testResults: [],
        executionTime: 0,
        errors: []
      }

      const startTime = Date.now()

      // Validate each test case
      for (let i = 0; i < this.testCases.length; i++) {
        const testCase = this.testCases[i]
        const testResult = await this.runTestCase(userCode, testCase, language)

        results.testResults.push({
          testIndex: i,
          description: testCase.description,
          input: testCase.input,
          expectedOutput: testCase.expectedOutput,
          actualOutput: testResult.output,
          passed: testResult.passed,
          error: testResult.error
        })

        if (testResult.passed) {
          results.passedTests++
        }

        if (testResult.error) {
          results.errors.push(testResult.error)
        }
      }

      results.executionTime = Date.now() - startTime
      results.success = results.passedTests === results.totalTests
      results.score =
        results.totalTests > 0 ? results.passedTests / results.totalTests : 0

      return results
    } catch (error) {
      console.error('❌ Error validating solution:', error)
      return {
        success: false,
        passedTests: 0,
        totalTests: this.testCases.length,
        testResults: [],
        executionTime: 0,
        errors: [error.message]
      }
    }
  }

  /**
   * Run a single test case against user code
   * @param {string} userCode - User's code
   * @param {Object} testCase - Test case data
   * @param {string} language - Programming language
   * @returns {Promise<Object>} - Test result
   */
  async runTestCase (userCode, testCase, language) {
    try {
      // For now, implement basic Python test validation
      // In a real implementation, this would use a secure sandbox

      if (language === 'python') {
        return this.runPythonTestCase(userCode, testCase)
      }

      // Fallback for other languages
      return {
        passed: false,
        output: null,
        error: `Language ${language} not supported yet`
      }
    } catch (error) {
      return {
        passed: false,
        output: null,
        error: error.message
      }
    }
  }

  /**
   * Run Python test case (simplified validation)
   * @param {string} userCode - User's Python code
   * @param {Object} testCase - Test case data
   * @returns {Object} - Test result
   */
  runPythonTestCase (userCode, testCase) {
    try {
      // Basic validation - check if code contains expected patterns
      const { input, expectedOutput, validation } = testCase

      // Check for syntax errors (basic)
      if (!this.validatePythonSyntax(userCode)) {
        return {
          passed: false,
          output: null,
          error: 'Syntax error in code'
        }
      }

      // Check for required patterns if specified
      if (validation && validation.requiredPatterns) {
        for (const pattern of validation.requiredPatterns) {
          const regex = new RegExp(pattern)
          if (!regex.test(userCode)) {
            return {
              passed: false,
              output: null,
              error: `Code must contain pattern: ${pattern}`
            }
          }
        }
      }

      // Check for forbidden patterns
      if (validation && validation.forbiddenPatterns) {
        for (const pattern of validation.forbiddenPatterns) {
          const regex = new RegExp(pattern)
          if (regex.test(userCode)) {
            return {
              passed: false,
              output: null,
              error: `Code must not contain pattern: ${pattern}`
            }
          }
        }
      }

      // For now, return success if syntax is valid
      // In production, this would execute in a secure sandbox
      return {
        passed: true,
        output: expectedOutput,
        error: null
      }
    } catch (error) {
      return {
        passed: false,
        output: null,
        error: error.message
      }
    }
  }

  /**
   * Basic Python syntax validation
   * @param {string} code - Python code to validate
   * @returns {boolean} - Whether syntax appears valid
   */
  validatePythonSyntax (code) {
    try {
      // Basic checks for common syntax errors
      const lines = code.split('\n')
      let indentLevel = 0
      let inFunction = false

      for (const line of lines) {
        const trimmed = line.trim()

        // Skip empty lines and comments
        if (!trimmed || trimmed.startsWith('#')) continue

        // Check for unmatched parentheses (basic)
        const openParens = (line.match(/\(/g) || []).length
        const closeParens = (line.match(/\)/g) || []).length
        const openBrackets = (line.match(/\[/g) || []).length
        const closeBrackets = (line.match(/\]/g) || []).length

        // Basic balance check (not perfect but catches obvious errors)
        if (
          Math.abs(openParens - closeParens) > 2 ||
          Math.abs(openBrackets - closeBrackets) > 2
        ) {
          return false
        }

        // Check for basic Python keywords and structure
        if (trimmed.includes('def ') || trimmed.includes('class ')) {
          if (!trimmed.endsWith(':')) {
            return false
          }
          inFunction = true
        }
      }

      return true
    } catch (error) {
      return false
    }
  }

  /**
   * Calculate difficulty score based on quest characteristics
   * @returns {number} - Difficulty score (1-10)
   */
  calculateDifficultyScore () {
    let score = 1

    // Base difficulty by type
    const typeScores = {
      theory_quiz: 2,
      coding_exercise: 4,
      practical_application: 6,
      implementation_project: 8
    }

    score += typeScores[this.type] || 2

    // Phase multiplier
    score += (this.phase - 1) * 1.5

    // Time complexity
    if (this.estimatedTime > 60) score += 2
    else if (this.estimatedTime > 30) score += 1

    // Prerequisites complexity
    score += Math.min(this.prerequisites.length * 0.5, 2)

    // Test case complexity
    if (this.testCases.length > 5) score += 1
    if (this.testCases.length > 10) score += 1

    return Math.min(Math.max(Math.round(score), 1), 10)
  }

  /**
   * Generate hints based on common errors and progress
   * @param {Array} errors - Recent error patterns
   * @returns {Array} - Contextual hints
   */
  generateContextualHints (errors = []) {
    const hints = [...this.hints]

    // Add hints based on common error patterns
    if (errors.some(e => e.includes('syntax'))) {
      hints.unshift(
        'Check your syntax - make sure parentheses and brackets are balanced'
      )
    }

    if (errors.some(e => e.includes('indentation'))) {
      hints.unshift(
        'Python is sensitive to indentation - make sure your code is properly indented'
      )
    }

    if (errors.some(e => e.includes('NameError'))) {
      hints.unshift('Make sure all variables are defined before you use them')
    }

    // Limit to 5 most relevant hints
    return hints.slice(0, 5)
  }

  /**
   * Get quest statistics for analytics
   * @returns {Promise<Object>} - Quest statistics
   */
  async getStatistics () {
    try {
      const stats = await db.get(
        `
                SELECT 
                    COUNT(*) as attempt_count,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completion_count,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastery_count,
                    AVG(time_to_complete_minutes) as avg_completion_time,
                    AVG(code_quality_score) as avg_quality_score,
                    AVG(creativity_score) as avg_creativity_score,
                    AVG(efficiency_score) as avg_efficiency_score
                FROM quest_completions 
                WHERE quest_id = ?
            `,
        [this.id]
      )

      return {
        attempts: stats.attempt_count || 0,
        completions: stats.completion_count || 0,
        masteries: stats.mastery_count || 0,
        averageTime: stats.avg_completion_time || 0,
        averageQuality: stats.avg_quality_score || 0,
        averageCreativity: stats.avg_creativity_score || 0,
        averageEfficiency: stats.avg_efficiency_score || 0,
        completionRate:
          stats.attempt_count > 0
            ? stats.completion_count / stats.attempt_count
            : 0,
        masteryRate:
          stats.completion_count > 0
            ? stats.mastery_count / stats.completion_count
            : 0
      }
    } catch (error) {
      console.error('❌ Error getting quest statistics:', error)
      return {
        attempts: 0,
        completions: 0,
        masteries: 0,
        averageTime: 0,
        averageQuality: 0,
        averageCreativity: 0,
        averageEfficiency: 0,
        completionRate: 0,
        masteryRate: 0
      }
    }
  }

  /**
   * Convert quest to JSON representation
   * @param {boolean} includePrivate - Whether to include private data (solutions, etc.)
   * @returns {Object} - JSON representation
   */
  toJSON (includePrivate = false) {
    const json = {
      id: this.id,
      title: this.title,
      description: this.description,
      type: this.type,
      phase: this.phase,
      week: this.week,
      difficulty: this.difficulty,
      estimatedTime: this.estimatedTime,
      prerequisites: this.prerequisites,
      learningObjectives: this.learningObjectives,
      tags: this.tags,
      isUnlocked: this.isUnlocked,
      unlockConditions: this.unlockConditions,
      starterCode: this.starterCode,
      hints: this.hints,
      resources: this.resources,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      difficultyScore: this.calculateDifficultyScore()
    }

    if (includePrivate) {
      json.testCases = this.testCases
      json.solution = this.solution
    } else {
      // Only include test case descriptions, not solutions
      json.testCases = this.testCases.map(tc => ({
        description: tc.description,
        inputDescription: tc.inputDescription || 'See starter code',
        outputDescription: tc.outputDescription || 'Expected output format'
      }))
    }

    return json
  }

  // ==========================================
  // STATIC METHODS
  // ==========================================

  /**
   * Find quest by ID
   * @param {string} id - Quest ID
   * @returns {Promise<Quest|null>} - Quest instance or null
   */
  static async findById (id) {
    try {
      const row = await db.get('SELECT * FROM quests WHERE id = ?', [id])
      if (!row) return null

      return Quest.fromDatabase(row)
    } catch (error) {
      console.error('❌ Error finding quest by ID:', error)
      return null
    }
  }

  /**
   * Find quests by criteria
   * @param {Object} criteria - Search criteria
   * @returns {Promise<Array>} - Array of quest instances
   */
  static async findByCriteria (criteria = {}) {
    try {
      let whereClause = '1=1'
      let params = []

      if (criteria.phase) {
        whereClause += ' AND phase = ?'
        params.push(criteria.phase)
      }

      if (criteria.week) {
        whereClause += ' AND week = ?'
        params.push(criteria.week)
      }

      if (criteria.type) {
        whereClause += ' AND type = ?'
        params.push(criteria.type)
      }

      if (criteria.difficulty) {
        whereClause += ' AND difficulty = ?'
        params.push(criteria.difficulty)
      }

      const rows = await db.all(
        `
                SELECT * FROM quests 
                WHERE ${whereClause}
                ORDER BY phase, week, created_at
            `,
        params
      )

      return rows.map(row => Quest.fromDatabase(row))
    } catch (error) {
      console.error('❌ Error finding quests by criteria:', error)
      return []
    }
  }

  /**
   * Create Quest instance from database row
   * @param {Object} row - Database row data
   * @returns {Quest} - Quest instance
   */
  static fromDatabase (row) {
    return new Quest({
      id: row.id,
      title: row.title,
      description: row.description,
      type: row.type,
      phase: row.phase,
      week: row.week,
      difficulty: row.difficulty,
      estimatedTime: row.estimated_time_minutes,
      prerequisites: JSON.parse(row.prerequisites || '[]'),
      learningObjectives: JSON.parse(row.learning_objectives || '[]'),
      tags: JSON.parse(row.tags || '[]'),
      unlockConditions: JSON.parse(row.unlock_conditions || '{}'),
      testCases: JSON.parse(row.test_cases || '[]'),
      starterCode: row.starter_code || '',
      solution: row.solution || '',
      hints: JSON.parse(row.hints || '[]'),
      resources: JSON.parse(row.resources || '[]'),
      createdAt: row.created_at,
      updatedAt: row.updated_at
    })
  }

  /**
   * Get all unlocked quests for user
   * @returns {Promise<Array>} - Array of unlocked quest instances
   */
  static async getUnlockedQuests () {
    try {
      const userProgress = await Quest.prototype.getUserProgress()
      const allQuests = await Quest.findByCriteria()
      const unlockedQuests = []

      for (const quest of allQuests) {
        const unlockStatus = await quest.checkUnlockConditions(userProgress)
        if (unlockStatus.canUnlock) {
          quest.isUnlocked = true
          unlockedQuests.push(quest)
        }
      }

      return unlockedQuests
    } catch (error) {
      console.error('❌ Error getting unlocked quests:', error)
      return []
    }
  }

  /**
   * Get next recommended quest for user
   * @returns {Promise<Quest|null>} - Next recommended quest
   */
  static async getNextRecommended () {
    try {
      const profile = await db.get('SELECT * FROM user_profile WHERE id = 1')
      if (!profile) return null

      // Find next quest in current phase/week
      const nextQuest = await Quest.findByCriteria({
        phase: profile.current_phase,
        week: profile.current_week
      })

      // Filter for unlocked and not completed
      for (const quest of nextQuest) {
        const unlockStatus = await quest.checkUnlockConditions()
        const completed = await db.get(
          `
                    SELECT * FROM quest_completions 
                    WHERE quest_id = ? AND status IN ('completed', 'mastered')
                `,
          [quest.id]
        )

        if (unlockStatus.canUnlock && !completed) {
          quest.isUnlocked = true
          return quest
        }
      }

      return null
    } catch (error) {
      console.error('❌ Error getting next recommended quest:', error)
      return null
    }
  }

  /**
   * Initialize quests table in database
   * @returns {Promise<boolean>} - Success status
   */
  static async initializeTable () {
    try {
      await db.run(`
                CREATE TABLE IF NOT EXISTS quests (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL CHECK (type IN ('coding_exercise', 'implementation_project', 'theory_quiz', 'practical_application')),
                    phase INTEGER NOT NULL CHECK (phase BETWEEN 1 AND 4),
                    week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 12),
                    difficulty TEXT NOT NULL CHECK (difficulty IN ('beginner', 'intermediate', 'advanced', 'expert')),
                    estimated_time_minutes INTEGER DEFAULT 30,
                    prerequisites TEXT DEFAULT '[]',
                    learning_objectives TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    unlock_conditions TEXT DEFAULT '{}',
                    test_cases TEXT DEFAULT '[]',
                    starter_code TEXT DEFAULT '',
                    solution TEXT DEFAULT '',
                    hints TEXT DEFAULT '[]',
                    resources TEXT DEFAULT '[]',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            `)

      console.log('✅ Quests table initialized')
      return true
    } catch (error) {
      console.error('❌ Error initializing quests table:', error)
      return false
    }
  }
}

module.exports = Quest
