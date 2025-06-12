-- Neural Odyssey Database Schema
-- SQLite database for personal ML learning companion
-- Designed for single-user local environment
-- 
-- Author: Neural Explorer
-- Created: 2025

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ==========================================
-- USER PROFILE AND SETTINGS
-- ==========================================

-- Single user profile (only one row needed)
CREATE TABLE user_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1), -- Enforce single user
    username TEXT NOT NULL DEFAULT 'Neural Explorer',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Learning preferences
    timezone TEXT DEFAULT 'UTC',
    preferred_session_length INTEGER DEFAULT 25, -- Pomodoro minutes
    daily_goal_minutes INTEGER DEFAULT 60,
    notification_enabled BOOLEAN DEFAULT 1,
    
    -- Current progress
    current_phase INTEGER DEFAULT 1,
    current_week INTEGER DEFAULT 1,
    current_lesson_id TEXT,
    
    -- Statistics
    total_study_minutes INTEGER DEFAULT 0,
    current_streak_days INTEGER DEFAULT 0,
    longest_streak_days INTEGER DEFAULT 0,
    last_activity_date DATE DEFAULT CURRENT_DATE,
    
    -- Gamification
    total_experience_points INTEGER DEFAULT 0,
    neural_explorer_level INTEGER DEFAULT 1,
    
    -- Preferences
    theme_preference TEXT DEFAULT 'neural-dark', -- neural-dark, neural-light, cosmic
    animation_enabled BOOLEAN DEFAULT 1,
    sound_enabled BOOLEAN DEFAULT 0
);

-- ==========================================
-- LEARNING PROGRESS TRACKING
-- ==========================================

-- Track progress through each lesson in the ML path
CREATE TABLE learning_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Lesson identification
    lesson_id TEXT NOT NULL UNIQUE, -- e.g., "linear_algebra_intro", "calculus_foundations"
    phase INTEGER NOT NULL, -- 1-4
    week INTEGER NOT NULL,  -- 1-12 per phase
    lesson_type TEXT NOT NULL CHECK (lesson_type IN ('theory', 'math', 'visual', 'coding')),
    lesson_title TEXT NOT NULL,
    
    -- Progress status
    status TEXT NOT NULL DEFAULT 'not_started' 
        CHECK (status IN ('not_started', 'in_progress', 'completed', 'mastered')),
    completion_percentage INTEGER DEFAULT 0 CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    
    -- Time tracking
    time_spent_minutes INTEGER DEFAULT 0,
    start_date DATE,
    completed_at DATETIME,
    mastered_at DATETIME,
    
    -- Learning quality metrics
    mastery_score REAL CHECK (mastery_score >= 0 AND mastery_score <= 1), -- 0.0-1.0
    confidence_level INTEGER CHECK (confidence_level >= 1 AND confidence_level <= 5), -- 1-5 scale
    difficulty_rating INTEGER CHECK (difficulty_rating >= 1 AND difficulty_rating <= 5), -- User's perceived difficulty
    
    -- Notes and reflections
    notes TEXT,
    key_insights TEXT,
    questions TEXT,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient queries
CREATE INDEX idx_learning_progress_phase_week ON learning_progress(phase, week);
CREATE INDEX idx_learning_progress_status ON learning_progress(status);
CREATE INDEX idx_learning_progress_updated ON learning_progress(updated_at);

-- ==========================================
-- QUEST AND PROJECT COMPLETIONS
-- ==========================================

-- Track coding exercises, implementation projects, and practical applications
CREATE TABLE quest_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Quest identification
    quest_id TEXT NOT NULL,
    quest_title TEXT NOT NULL,
    quest_type TEXT NOT NULL CHECK (quest_type IN ('coding_exercise', 'implementation_project', 'theory_quiz', 'practical_application')),
    phase INTEGER NOT NULL,
    week INTEGER NOT NULL,
    
    -- Related lesson
    related_lesson_id TEXT,
    
    -- Completion details
    status TEXT NOT NULL DEFAULT 'attempted' 
        CHECK (status IN ('attempted', 'completed', 'mastered')),
    attempts_count INTEGER DEFAULT 1,
    time_to_complete_minutes INTEGER,
    
    -- Code and solutions
    user_code TEXT, -- User's submitted code
    final_solution TEXT, -- Final working solution
    test_results TEXT, -- JSON string of test case results
    
    -- Quality metrics
    code_quality_score INTEGER CHECK (code_quality_score >= 1 AND code_quality_score <= 5),
    creativity_score INTEGER CHECK (creativity_score >= 1 AND creativity_score <= 5),
    efficiency_score INTEGER CHECK (efficiency_score >= 1 AND efficiency_score <= 5),
    
    -- Learning artifacts
    approach_description TEXT, -- How they approached the problem
    learned_concepts TEXT, -- What they learned
    challenges_faced TEXT, -- Difficulties encountered
    
    -- Timestamps
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    
    FOREIGN KEY (related_lesson_id) REFERENCES learning_progress(lesson_id)
);

CREATE INDEX idx_quest_completions_phase_week ON quest_completions(phase, week);
CREATE INDEX idx_quest_completions_status ON quest_completions(status);
CREATE INDEX idx_quest_completions_type ON quest_completions(quest_type);

-- ==========================================
-- DAILY LEARNING SESSIONS
-- ==========================================

-- Track daily Pomodoro-style learning sessions with rotation
CREATE TABLE daily_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Session identification
    session_date DATE NOT NULL,
    session_type TEXT NOT NULL CHECK (session_type IN ('math', 'coding', 'visual_projects', 'real_applications')),
    session_number INTEGER DEFAULT 1, -- Multiple sessions per day
    
    -- Session timing
    start_time DATETIME,
    end_time DATETIME,
    planned_duration_minutes INTEGER DEFAULT 25,
    actual_duration_minutes INTEGER,
    break_duration_minutes INTEGER DEFAULT 5,
    
    -- Session quality
    focus_score INTEGER CHECK (focus_score >= 1 AND focus_score <= 10), -- 1-10 scale
    energy_level INTEGER CHECK (energy_level >= 1 AND energy_level <= 10), -- Before session
    mood_before TEXT, -- 'excited', 'tired', 'focused', 'distracted', etc.
    mood_after TEXT,
    
    -- Session content
    topics_covered TEXT, -- JSON array of topics
    goals TEXT, -- What they planned to accomplish
    completed_goals TEXT, -- What they actually completed
    session_notes TEXT,
    
    -- Distractions and interruptions
    interruption_count INTEGER DEFAULT 0,
    distraction_types TEXT, -- JSON array: ['phone', 'email', 'noise', etc.]
    
    -- Next session planning
    next_session_type TEXT CHECK (next_session_type IN ('math', 'coding', 'visual_projects', 'real_applications')),
    next_session_planned_at DATETIME,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_daily_sessions_date ON daily_sessions(session_date);
CREATE INDEX idx_daily_sessions_type ON daily_sessions(session_type);

-- ==========================================
-- SKILL POINTS AND ACHIEVEMENTS
-- ==========================================

-- Track skill points earned across different categories
CREATE TABLE skill_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Skill category
    category TEXT NOT NULL CHECK (category IN ('mathematics', 'programming', 'theory', 'applications', 'creativity', 'persistence')),
    
    -- Points
    points_earned INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL, -- Why points were earned
    multiplier REAL DEFAULT 1.0, -- Bonus multipliers
    
    -- Source
    source_type TEXT CHECK (source_type IN ('lesson_completion', 'quest_completion', 'milestone', 'streak', 'exploration')),
    related_quest_id TEXT,
    related_lesson_id TEXT,
    
    -- Timestamps
    earned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (related_quest_id) REFERENCES quest_completions(quest_id),
    FOREIGN KEY (related_lesson_id) REFERENCES learning_progress(lesson_id)
);

CREATE INDEX idx_skill_points_category ON skill_points(category);
CREATE INDEX idx_skill_points_earned_at ON skill_points(earned_at);

-- ==========================================
-- NEURAL VAULT SYSTEM
-- ==========================================

-- Track unlocked vault items (rewards)
CREATE TABLE vault_unlocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Vault item identification
    vault_item_id TEXT NOT NULL UNIQUE, -- Links to vault-items.json
    category TEXT NOT NULL CHECK (category IN ('secret_archives', 'controversy_files', 'beautiful_mind')),
    
    -- Unlock details
    unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    unlock_trigger TEXT NOT NULL, -- What triggered the unlock
    unlock_phase INTEGER,
    unlock_week INTEGER,
    unlock_lesson_id TEXT,
    
    -- User interaction
    is_read BOOLEAN DEFAULT 0,
    read_at DATETIME,
    time_spent_reading_minutes INTEGER DEFAULT 0,
    
    -- User feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5), -- 1-5 stars
    user_notes TEXT, -- Personal thoughts and connections
    is_favorite BOOLEAN DEFAULT 0,
    
    -- Social features (for future)
    share_count INTEGER DEFAULT 0,
    
    FOREIGN KEY (unlock_lesson_id) REFERENCES learning_progress(lesson_id)
);

CREATE INDEX idx_vault_unlocks_category ON vault_unlocks(category);
CREATE INDEX idx_vault_unlocks_unlocked_at ON vault_unlocks(unlocked_at);
CREATE INDEX idx_vault_unlocks_is_read ON vault_unlocks(is_read);

-- ==========================================
-- SPACED REPETITION SYSTEM
-- ==========================================

-- Track concepts for spaced repetition review
CREATE TABLE spaced_repetition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Concept identification
    concept_id TEXT NOT NULL UNIQUE, -- e.g., "eigenvalues_definition", "gradient_descent_intuition"
    concept_title TEXT NOT NULL,
    concept_type TEXT CHECK (concept_type IN ('definition', 'formula', 'intuition', 'application', 'connection')),
    related_lesson_id TEXT,
    
    -- Spaced repetition algorithm (SM-2)
    easiness_factor REAL DEFAULT 2.5 CHECK (easiness_factor >= 1.3),
    interval_days INTEGER DEFAULT 1,
    repetitions INTEGER DEFAULT 0,
    
    -- Review scheduling
    last_review_date DATE,
    next_review_date DATE NOT NULL,
    
    -- Performance tracking
    total_reviews INTEGER DEFAULT 0,
    correct_reviews INTEGER DEFAULT 0,
    accuracy_rate REAL DEFAULT 0.0,
    
    -- Content
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    hint_text TEXT,
    tags TEXT, -- JSON array of tags
    
    -- Status
    is_active BOOLEAN DEFAULT 1,
    is_mastered BOOLEAN DEFAULT 0,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (related_lesson_id) REFERENCES learning_progress(lesson_id)
);

CREATE INDEX idx_spaced_repetition_next_review ON spaced_repetition(next_review_date);
CREATE INDEX idx_spaced_repetition_concept_type ON spaced_repetition(concept_type);
CREATE INDEX idx_spaced_repetition_is_active ON spaced_repetition(is_active);

-- ==========================================
-- KNOWLEDGE GRAPH
-- ==========================================

-- Track connections between concepts for building knowledge maps
CREATE TABLE knowledge_connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Connection details
    source_concept_id TEXT NOT NULL,
    target_concept_id TEXT NOT NULL,
    connection_type TEXT NOT NULL CHECK (connection_type IN ('prerequisite', 'builds_on', 'related_to', 'contrasts_with', 'applies_to')),
    
    -- Strength and confidence
    connection_strength REAL DEFAULT 0.5 CHECK (connection_strength >= 0 AND connection_strength <= 1),
    user_confidence INTEGER CHECK (user_confidence >= 1 AND user_confidence <= 5),
    
    -- Learning context
    discovered_during_lesson TEXT,
    discovery_method TEXT CHECK (discovery_method IN ('explicit_teaching', 'user_insight', 'pattern_recognition', 'problem_solving')),
    
    -- Description
    connection_description TEXT,
    example_or_proof TEXT,
    
    -- Metadata
    is_verified BOOLEAN DEFAULT 0, -- Whether connection has been validated
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (discovered_during_lesson) REFERENCES learning_progress(lesson_id),
    UNIQUE(source_concept_id, target_concept_id, connection_type)
);

CREATE INDEX idx_knowledge_connections_source ON knowledge_connections(source_concept_id);
CREATE INDEX idx_knowledge_connections_target ON knowledge_connections(target_concept_id);
CREATE INDEX idx_knowledge_connections_type ON knowledge_connections(connection_type);

-- ==========================================
-- LEARNING ANALYTICS
-- ==========================================

-- Track learning patterns and insights for optimization
CREATE TABLE learning_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Time period
    analysis_date DATE NOT NULL,
    period_type TEXT NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly')),
    
    -- Learning metrics
    total_study_minutes INTEGER DEFAULT 0,
    lessons_completed INTEGER DEFAULT 0,
    quests_completed INTEGER DEFAULT 0,
    vault_items_unlocked INTEGER DEFAULT 0,
    
    -- Quality metrics
    average_focus_score REAL,
    average_energy_level REAL,
    concepts_mastered INTEGER DEFAULT 0,
    concepts_reviewed INTEGER DEFAULT 0,
    
    -- Patterns
    most_productive_hour INTEGER, -- 0-23
    most_productive_day_of_week INTEGER, -- 1-7 (Monday = 1)
    preferred_session_type TEXT,
    
    -- Goal tracking
    daily_goal_achievement_rate REAL DEFAULT 0.0, -- 0.0-1.0
    streak_maintained BOOLEAN DEFAULT 0,
    
    -- Insights (JSON)
    insights TEXT, -- JSON object with discovered patterns
    recommendations TEXT, -- JSON array of recommendations
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_learning_analytics_date ON learning_analytics(analysis_date);
CREATE INDEX idx_learning_analytics_period ON learning_analytics(period_type);

-- ==========================================
-- TRIGGERS FOR AUTOMATED UPDATES
-- ==========================================

-- Update user profile last activity when learning progress changes
CREATE TRIGGER update_user_activity_on_progress
    AFTER UPDATE ON learning_progress
    WHEN NEW.updated_at != OLD.updated_at
BEGIN
    UPDATE user_profile 
    SET last_activity_date = CURRENT_DATE,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;

-- Update user profile total study time when session ends
CREATE TRIGGER update_total_study_time
    AFTER UPDATE ON daily_sessions
    WHEN NEW.actual_duration_minutes IS NOT NULL AND OLD.actual_duration_minutes IS NULL
BEGIN
    UPDATE user_profile 
    SET total_study_minutes = total_study_minutes + NEW.actual_duration_minutes,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;

-- Auto-update learning progress updated_at timestamp
CREATE TRIGGER update_learning_progress_timestamp
    AFTER UPDATE ON learning_progress
BEGIN
    UPDATE learning_progress 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Auto-update spaced repetition timestamp
CREATE TRIGGER update_spaced_repetition_timestamp
    AFTER UPDATE ON spaced_repetition
BEGIN
    UPDATE spaced_repetition 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- ==========================================
-- VIEWS FOR COMMON QUERIES
-- ==========================================

-- Current progress overview
CREATE VIEW current_progress_overview AS
SELECT 
    up.current_phase,
    up.current_week,
    up.total_study_minutes,
    up.current_streak_days,
    COUNT(lp.id) as total_lessons,
    COUNT(CASE WHEN lp.status = 'completed' THEN 1 END) as completed_lessons,
    COUNT(CASE WHEN lp.status = 'mastered' THEN 1 END) as mastered_lessons,
    COUNT(qc.id) as total_quests,
    COUNT(CASE WHEN qc.status = 'completed' THEN 1 END) as completed_quests,
    COUNT(vu.id) as vault_items_unlocked
FROM user_profile up
LEFT JOIN learning_progress lp ON 1=1
LEFT JOIN quest_completions qc ON 1=1  
LEFT JOIN vault_unlocks vu ON 1=1
WHERE up.id = 1
GROUP BY up.id;

-- Weekly learning summary
CREATE VIEW weekly_learning_summary AS
SELECT 
    lp.phase,
    lp.week,
    COUNT(*) as total_lessons_in_week,
    COUNT(CASE WHEN lp.status IN ('completed', 'mastered') THEN 1 END) as completed_lessons,
    SUM(lp.time_spent_minutes) as total_time_spent,
    AVG(lp.mastery_score) as average_mastery_score,
    COUNT(CASE WHEN lp.status = 'mastered' THEN 1 END) as mastered_lessons
FROM learning_progress lp
GROUP BY lp.phase, lp.week
ORDER BY lp.phase, lp.week;

-- Skill points summary
CREATE VIEW skill_points_summary AS
SELECT 
    category,
    SUM(points_earned) as total_points,
    COUNT(*) as times_earned,
    AVG(points_earned) as average_points_per_earn,
    MAX(points_earned) as highest_single_earn,
    MAX(earned_at) as last_earned
FROM skill_points
GROUP BY category
ORDER BY total_points DESC;

-- Vault progress by category
CREATE VIEW vault_progress_by_category AS
SELECT 
    category,
    COUNT(*) as total_unlocked,
    COUNT(CASE WHEN is_read = 1 THEN 1 END) as total_read,
    AVG(user_rating) as average_rating,
    COUNT(CASE WHEN is_favorite = 1 THEN 1 END) as favorites_count
FROM vault_unlocks
GROUP BY category;

-- ==========================================
-- INITIAL DATA POPULATION
-- ==========================================

-- Insert default user profile
INSERT INTO user_profile (id, username) VALUES (1, 'Neural Explorer');

-- ==========================================
-- DATABASE MAINTENANCE
-- ==========================================

-- Optimize database performance
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA temp_store = memory;
PRAGMA mmap_size = 268435456;

-- Analyze tables for query optimization
ANALYZE;