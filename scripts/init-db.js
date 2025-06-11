#!/usr/bin/env node

/**
 * Neural Odyssey Database Initialization Script
 * 
 * This script:
 * 1. Creates the SQLite database with the schema
 * 2. Populates initial learning content from the ML path
 * 3. Sets up vault items from vault-items.json
 * 4. Creates sample data for development
 * 
 * Run: node scripts/init-db.js
 */

const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const path = require('path');

// File paths
const DB_PATH = path.join(__dirname, '../data/user-progress.sqlite');
const SCHEMA_PATH = path.join(__dirname, '../data/schema.sql');
const VAULT_ITEMS_PATH = path.join(__dirname, '../data/vault-items.json');
const DATA_DIR = path.join(__dirname, '../data');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    console.log('✅ Created data directory');
}

// Learning content structure based on ML path
const ML_LEARNING_PATH = {
    phase1: {
        title: "Mathematical Foundations and Historical Context",
        duration: "Months 1-3",
        weeks: [
            {
                week: 1,
                title: "Linear Algebra Through the Lens of Data",
                lessons: [
                    { id: "linear_algebra_intro", title: "Introduction to Linear Algebra", type: "theory" },
                    { id: "vectors_and_matrices", title: "Vectors and Matrices", type: "math" },
                    { id: "geometric_transformations", title: "Geometric Transformations", type: "visual" },
                    { id: "matrix_operations_implementation", title: "Matrix Operations from Scratch", type: "coding" }
                ]
            },
            {
                week: 2,
                title: "Linear Algebra Continued",
                lessons: [
                    { id: "eigenvalues_eigenvectors", title: "Eigenvalues and Eigenvectors", type: "math" },
                    { id: "linear_algebra_eigenvalues", title: "Eigenvalues in Machine Learning", type: "theory" },
                    { id: "pca_foundations", title: "Principal Component Analysis Foundations", type: "visual" },
                    { id: "linear_algebra_project", title: "Build a Data Transformation Tool", type: "coding" }
                ]
            },
            {
                week: 3,
                title: "Calculus as the Engine of Learning",
                lessons: [
                    { id: "calculus_foundations", title: "Calculus Foundations for ML", type: "theory" },
                    { id: "derivatives_geometric", title: "Derivatives and Geometric Intuition", type: "math" },
                    { id: "gradient_visualization", title: "Gradient Visualization", type: "visual" },
                    { id: "gradient_descent_implementation", title: "Gradient Descent from Scratch", type: "coding" }
                ]
            },
            {
                week: 4,
                title: "Chain Rule and Backpropagation",
                lessons: [
                    { id: "chain_rule_theory", title: "The Chain Rule in Detail", type: "math" },
                    { id: "chain_rule_implementation", title: "Chain Rule Implementation", type: "coding" },
                    { id: "backprop_visualization", title: "Backpropagation Visualization", type: "visual" },
                    { id: "neural_network_basic", title: "Build Your First Neural Network", type: "coding" }
                ]
            },
            {
                week: 5,
                title: "Statistics and Probability",
                lessons: [
                    { id: "probability_foundations", title: "Probability Foundations", type: "theory" },
                    { id: "bayesian_thinking", title: "Bayesian Thinking", type: "math" },
                    { id: "distributions_visualization", title: "Distribution Visualizations", type: "visual" },
                    { id: "statistical_analysis_project", title: "Statistical Analysis Project", type: "coding" }
                ]
            },
            {
                week: 6,
                title: "Statistics for ML",
                lessons: [
                    { id: "statistical_learning_theory", title: "Statistical Learning Theory", type: "theory" },
                    { id: "mathematical_foundations_synthesis", title: "Mathematical Foundations Synthesis", type: "math" },
                    { id: "probability_distributions", title: "Probability Distributions in ML", type: "visual" },
                    { id: "stats_ml_project", title: "Statistics + ML Combined Project", type: "coding" }
                ]
            },
            {
                week: 7,
                title: "Historical Context - Early AI",
                lessons: [
                    { id: "mcculloch_pitts_neuron", title: "McCulloch-Pitts Neuron", type: "theory" },
                    { id: "perceptron_history", title: "Perceptron and Early Learning", type: "theory" },
                    { id: "early_ai_timeline", title: "Early AI Timeline Visualization", type: "visual" },
                    { id: "perceptron_implementation", title: "Implement Historical Perceptron", type: "coding" }
                ]
            },
            {
                week: 8,
                title: "AI Winters and Limitations",
                lessons: [
                    { id: "perceptron_limitations", title: "Perceptron Limitations and XOR Problem", type: "theory" },
                    { id: "ai_winter_history", title: "AI Winter Historical Analysis", type: "theory" },
                    { id: "xor_problem_visualization", title: "XOR Problem Visualization", type: "visual" },
                    { id: "multilayer_perceptron", title: "Build Multilayer Perceptron", type: "coding" }
                ]
            },
            {
                week: 9,
                title: "Building First Learning Systems",
                lessons: [
                    { id: "linear_regression_theory", title: "Linear Regression Theory", type: "theory" },
                    { id: "linear_regression_math", title: "Linear Regression Mathematics", type: "math" },
                    { id: "regression_visualization", title: "Regression Line Visualization", type: "visual" },
                    { id: "linear_regression_scratch", title: "Linear Regression from Scratch", type: "coding" }
                ]
            },
            {
                week: 10,
                title: "Logistic Regression",
                lessons: [
                    { id: "logistic_regression_theory", title: "Logistic Regression Theory", type: "theory" },
                    { id: "sigmoid_function", title: "Sigmoid Function and Probability", type: "math" },
                    { id: "classification_boundaries", title: "Classification Boundary Visualization", type: "visual" },
                    { id: "logistic_regression_implementation", title: "Logistic Regression Implementation", type: "coding" }
                ]
            },
            {
                week: 11,
                title: "K-Means Clustering",
                lessons: [
                    { id: "clustering_theory", title: "Unsupervised Learning and Clustering", type: "theory" },
                    { id: "kmeans_algorithm", title: "K-Means Algorithm Mathematics", type: "math" },
                    { id: "clustering_visualization", title: "Interactive Clustering Visualization", type: "visual" },
                    { id: "kmeans_implementation", title: "K-Means from Scratch", type: "coding" }
                ]
            },
            {
                week: 12,
                title: "Phase 1 Integration",
                lessons: [
                    { id: "foundations_integration", title: "Mathematical Foundations Integration", type: "theory" },
                    { id: "concept_connections", title: "Connecting All Mathematical Concepts", type: "math" },
                    { id: "foundations_portfolio", title: "Create Your Foundations Portfolio", type: "visual" },
                    { id: "phase1_capstone_project", title: "Phase 1 Capstone: End-to-End ML Pipeline", type: "coding" }
                ]
            }
        ]
    },
    phase2: {
        title: "Core Machine Learning with Deep Understanding",
        duration: "Months 4-6",
        weeks: [
            // Phase 2 weeks 13-24 would be defined here
            // For brevity, adding just the structure for now
        ]
    },
    phase3: {
        title: "Advanced Topics and Modern AI",
        duration: "Months 7-9", 
        weeks: [
            // Phase 3 weeks 25-36 would be defined here
        ]
    },
    phase4: {
        title: "Mastery and Innovation",
        duration: "Months 10-12",
        weeks: [
            // Phase 4 weeks 37-48 would be defined here
        ]
    }
};

async function createDatabase() {
    return new Promise((resolve, reject) => {
        // Remove existing database
        if (fs.existsSync(DB_PATH)) {
            fs.unlinkSync(DB_PATH);
            console.log('🗑️  Removed existing database');
        }

        const db = new sqlite3.Database(DB_PATH, (err) => {
            if (err) {
                reject(err);
                return;
            }
            console.log('✅ Connected to SQLite database');
            resolve(db);
        });
    });
}

async function createTables(db) {
    return new Promise((resolve, reject) => {
        // Read schema file we created earlier
        const schemaSQL = `
            -- User Profile and Settings
            CREATE TABLE user_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL DEFAULT 'Neural Explorer',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                timezone TEXT DEFAULT 'UTC',
                preferred_session_length INTEGER DEFAULT 25,
                daily_goal_minutes INTEGER DEFAULT 60,
                current_phase INTEGER DEFAULT 1,
                current_week INTEGER DEFAULT 1,
                total_study_minutes INTEGER DEFAULT 0,
                current_streak_days INTEGER DEFAULT 0,
                longest_streak_days INTEGER DEFAULT 0,
                last_activity_date DATE,
                avatar_style TEXT DEFAULT 'neural_network',
                theme_preference TEXT DEFAULT 'dark'
            );

            CREATE TABLE learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase INTEGER NOT NULL,
                week INTEGER NOT NULL,
                lesson_id TEXT NOT NULL,
                lesson_title TEXT NOT NULL,
                lesson_type TEXT CHECK(lesson_type IN ('theory', 'math', 'visual', 'coding')) DEFAULT 'theory',
                status TEXT CHECK(status IN ('not_started', 'in_progress', 'completed', 'mastered')) DEFAULT 'not_started',
                completion_percentage INTEGER DEFAULT 0,
                time_spent_minutes INTEGER DEFAULT 0,
                completed_at DATETIME,
                mastery_score REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phase, week, lesson_id)
            );

            CREATE TABLE vault_unlocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vault_item_id TEXT NOT NULL,
                category TEXT CHECK(category IN ('secret_archives', 'controversy_files', 'beautiful_mind')) NOT NULL,
                unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_read BOOLEAN DEFAULT FALSE,
                read_at DATETIME,
                user_rating INTEGER CHECK(user_rating BETWEEN 1 AND 5),
                user_notes TEXT,
                UNIQUE(vault_item_id)
            );

            CREATE TABLE daily_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date DATE NOT NULL,
                session_type TEXT CHECK(session_type IN ('math', 'coding', 'visual_projects', 'real_applications')) NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration_minutes INTEGER,
                focus_score INTEGER CHECK(focus_score BETWEEN 1 AND 10),
                energy_level INTEGER CHECK(energy_level BETWEEN 1 AND 10),
                completed_goals TEXT,
                session_notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE quest_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quest_id TEXT NOT NULL,
                quest_title TEXT NOT NULL,
                quest_type TEXT CHECK(quest_type IN ('coding_exercise', 'implementation_project', 'theory_quiz', 'practical_application')) NOT NULL,
                phase INTEGER NOT NULL,
                week INTEGER NOT NULL,
                difficulty_level INTEGER CHECK(difficulty_level BETWEEN 1 AND 5) NOT NULL,
                status TEXT CHECK(status IN ('attempted', 'completed', 'mastered')) DEFAULT 'attempted',
                code_solution TEXT,
                execution_result TEXT,
                time_to_complete_minutes INTEGER,
                attempts_count INTEGER DEFAULT 1,
                hint_used_count INTEGER DEFAULT 0,
                completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                mentor_feedback TEXT,
                self_reflection TEXT
            );

            CREATE TABLE skill_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT CHECK(category IN ('mathematics', 'programming', 'theory', 'applications', 'creativity', 'persistence')) NOT NULL,
                points_earned INTEGER NOT NULL,
                reason TEXT NOT NULL,
                earned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                related_quest_id TEXT,
                related_lesson_id TEXT
            );

            CREATE TABLE knowledge_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_a TEXT NOT NULL,
                concept_b TEXT NOT NULL,
                connection_type TEXT CHECK(connection_type IN ('builds_on', 'applies_to', 'similar_to', 'contrasts_with', 'enables')) NOT NULL,
                strength REAL CHECK(strength BETWEEN 0.0 AND 1.0) DEFAULT 0.5,
                discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_reinforced_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_explanation TEXT,
                UNIQUE(concept_a, concept_b, connection_type)
            );

            CREATE TABLE learning_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                measurement_date DATE NOT NULL,
                category TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE spaced_repetition (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT NOT NULL,
                concept_title TEXT NOT NULL,
                difficulty_factor REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 1,
                repetition_count INTEGER DEFAULT 0,
                next_review_date DATE NOT NULL,
                last_reviewed_at DATETIME,
                quality_score INTEGER CHECK(quality_score BETWEEN 0 AND 5),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(concept_id)
            );

            CREATE TABLE backup_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                backup_type TEXT CHECK(backup_type IN ('daily', 'weekly', 'manual')) NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                checksum TEXT,
                restoration_tested BOOLEAN DEFAULT FALSE
            );

            -- Create indexes
            CREATE INDEX idx_learning_progress_phase_week ON learning_progress(phase, week);
            CREATE INDEX idx_learning_progress_status ON learning_progress(status);
            CREATE INDEX idx_vault_unlocks_category ON vault_unlocks(category);
            CREATE INDEX idx_daily_sessions_date ON daily_sessions(session_date);
            CREATE INDEX idx_quest_completions_phase_week ON quest_completions(phase, week);
            CREATE INDEX idx_spaced_repetition_next_review ON spaced_repetition(next_review_date);

            -- Insert default user profile
            INSERT INTO user_profile (username) VALUES ('Neural Explorer');
        `;

        db.exec(schemaSQL, (err) => {
            if (err) {
                reject(err);
                return;
            }
            console.log('✅ Created database tables');
            resolve();
        });
    });
}

async function populateLearningContent(db) {
    return new Promise((resolve, reject) => {
        console.log('📚 Populating learning content...');
        
        const stmt = db.prepare(`
            INSERT INTO learning_progress (phase, week, lesson_id, lesson_title, lesson_type)
            VALUES (?, ?, ?, ?, ?)
        `);

        let totalLessons = 0;

        // Populate Phase 1 content
        ML_LEARNING_PATH.phase1.weeks.forEach((weekData, weekIndex) => {
            const weekNumber = weekIndex + 1;
            weekData.lessons.forEach((lesson) => {
                stmt.run(1, weekNumber, lesson.id, lesson.title, lesson.type);
                totalLessons++;
            });
        });

        // Add placeholder lessons for other phases
        // Phase 2: weeks 13-24
        for (let week = 13; week <= 24; week++) {
            for (let i = 1; i <= 4; i++) {
                stmt.run(2, week, `phase2_week${week}_lesson${i}`, `Phase 2 Week ${week} Lesson ${i}`, ['theory', 'math', 'visual', 'coding'][i-1]);
                totalLessons++;
            }
        }

        // Phase 3: weeks 25-36  
        for (let week = 25; week <= 36; week++) {
            for (let i = 1; i <= 4; i++) {
                stmt.run(3, week, `phase3_week${week}_lesson${i}`, `Phase 3 Week ${week} Lesson ${i}`, ['theory', 'math', 'visual', 'coding'][i-1]);
                totalLessons++;
            }
        }

        // Phase 4: weeks 37-48
        for (let week = 37; week <= 48; week++) {
            for (let i = 1; i <= 4; i++) {
                stmt.run(4, week, `phase4_week${week}_lesson${i}`, `Phase 4 Week ${week} Lesson ${i}`, ['theory', 'math', 'visual', 'coding'][i-1]);
                totalLessons++;
            }
        }

        stmt.finalize((err) => {
            if (err) {
                reject(err);
                return;
            }
            console.log(`✅ Added ${totalLessons} lessons to learning path`);
            resolve();
        });
    });
}

async function populateVaultItems(db) {
    return new Promise((resolve, reject) => {
        console.log('🗝️  Setting up vault items...');
        
        // Read vault items JSON
        if (!fs.existsSync(VAULT_ITEMS_PATH)) {
            console.log('⚠️  vault-items.json not found, skipping vault setup');
            resolve();
            return;
        }

        const vaultData = JSON.parse(fs.readFileSync(VAULT_ITEMS_PATH, 'utf8'));
        
        // We don't pre-populate vault_unlocks table - items are unlocked as user progresses
        // But we can add some initial skill points and sample data for development
        
        const stmt = db.prepare(`
            INSERT INTO skill_points (category, points_earned, reason)
            VALUES (?, ?, ?)
        `);

        // Add some starting skill points
        stmt.run('persistence', 50, 'Started your Neural Odyssey journey!');
        stmt.run('mathematics', 10, 'First database initialization');
        
        stmt.finalize((err) => {
            if (err) {
                reject(err);
                return;
            }
            console.log(`✅ Vault system ready with ${Object.keys(vaultData.secretArchives).length + Object.keys(vaultData.controversyFiles).length + Object.keys(vaultData.beautifulMindCollection).length} rewards`);
            resolve();
        });
    });
}

async function addSampleData(db) {
    return new Promise((resolve, reject) => {
        console.log('🎯 Adding sample data for development...');
        
        // Add a sample session
        db.run(`
            INSERT INTO daily_sessions (session_date, session_type, start_time, end_time, duration_minutes, focus_score, energy_level, session_notes)
            VALUES (date('now'), 'math', datetime('now', '-2 hours'), datetime('now', '-1 hour'), 60, 8, 7, 'Great first session learning about linear algebra!')
        `);

        // Add sample spaced repetition items
        const concepts = [
            'Matrix multiplication',
            'Eigenvalues and eigenvectors', 
            'Gradient descent',
            'Chain rule',
            'Backpropagation'
        ];

        const stmt = db.prepare(`
            INSERT INTO spaced_repetition (concept_id, concept_title, next_review_date)
            VALUES (?, ?, date('now', '+1 day'))
        `);

        concepts.forEach((concept, index) => {
            stmt.run(`concept_${index + 1}`, concept);
        });

        stmt.finalize((err) => {
            if (err) {
                reject(err);
                return;
            }
            console.log('✅ Added sample development data');
            resolve();
        });
    });
}

async function main() {
    try {
        console.log('🚀 Initializing Neural Odyssey database...\n');
        
        const db = await createDatabase();
        await createTables(db);
        await populateLearningContent(db);
        await populateVaultItems(db);
        await addSampleData(db);
        
        // Close database
        db.close((err) => {
            if (err) {
                console.error('❌ Error closing database:', err);
                return;
            }
            console.log('\n🎉 Database initialization complete!');
            console.log(`📍 Database location: ${DB_PATH}`);
            console.log('🎯 Ready to start your Neural Odyssey!\n');
        });
        
    } catch (error) {
        console.error('❌ Database initialization failed:', error);
        process.exit(1);
    }
}

// Run the initialization
if (require.main === module) {
    main();
}

module.exports = { createDatabase, createTables, populateLearningContent };