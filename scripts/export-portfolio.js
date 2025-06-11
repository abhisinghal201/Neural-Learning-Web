#!/usr/bin/env node

/**
 * Neural Odyssey Portfolio Export Script
 * 
 * This script exports your completed projects and learning progress to:
 * - Markdown files for easy sharing
 * - JSON files for data backup
 * - Portfolio websites for showcasing
 * 
 * Usage:
 *   node scripts/export-portfolio.js [options]
 * 
 * Options:
 *   --format=md|json|html|all    Export format (default: md)
 *   --phase=1|2|3|4|all          Specific phase or all (default: all)
 *   --type=completed|mastered    Progress level (default: completed)
 *   --output=path                Output directory (default: exports/)
 *   --template=minimal|detailed  Template style (default: detailed)
 */

const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

// Command line argument parsing
const args = process.argv.slice(2);
const options = {
    format: 'md',
    phase: 'all',
    type: 'completed',
    output: 'exports',
    template: 'detailed'
};

// Parse command line arguments
args.forEach(arg => {
    if (arg.startsWith('--')) {
        const [key, value] = arg.substring(2).split('=');
        if (value) options[key] = value;
    }
});

// File paths
const DB_PATH = path.join(__dirname, '../data/user-progress.sqlite');
const OUTPUT_DIR = path.join(__dirname, '..', options.output);
const TEMPLATES_DIR = path.join(__dirname, '../docs/templates');

class PortfolioExporter {
    constructor(dbPath) {
        this.dbPath = dbPath;
        this.db = null;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database(this.dbPath, sqlite3.OPEN_READONLY, (err) => {
                if (err) {
                    reject(new Error(`Failed to connect to database: ${err.message}`));
                    return;
                }
                console.log('✅ Connected to database');
                resolve();
            });
        });
    }

    async close() {
        return new Promise((resolve) => {
            if (this.db) {
                this.db.close(() => {
                    console.log('✅ Database connection closed');
                    resolve();
                });
            } else {
                resolve();
            }
        });
    }

    async getUserProfile() {
        return new Promise((resolve, reject) => {
            this.db.get(`
                SELECT * FROM user_profile WHERE id = 1
            `, (err, row) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(row);
            });
        });
    }

    async getCompletedLessons(phase = 'all', minStatus = 'completed') {
        const phaseCondition = phase === 'all' ? '' : `AND phase = ${phase}`;
        const statusCondition = minStatus === 'completed' 
            ? "AND status IN ('completed', 'mastered')"
            : "AND status = 'mastered'";

        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT * FROM learning_progress 
                WHERE 1=1 ${phaseCondition} ${statusCondition}
                ORDER BY phase, week, lesson_id
            `, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(rows);
            });
        });
    }

    async getCompletedQuests(phase = 'all') {
        const phaseCondition = phase === 'all' ? '' : `AND phase = ${phase}`;

        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT * FROM quest_completions 
                WHERE status IN ('completed', 'mastered') ${phaseCondition}
                ORDER BY phase, week, completed_at
            `, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(rows);
            });
        });
    }

    async getSkillPoints() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT category, SUM(points_earned) as total_points, COUNT(*) as achievements
                FROM skill_points 
                GROUP BY category
                ORDER BY total_points DESC
            `, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(rows);
            });
        });
    }

    async getUnlockedVaultItems() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT * FROM vault_unlocks 
                ORDER BY unlocked_at DESC
            `, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(rows);
            });
        });
    }

    async getLearningAnalytics() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT 
                    COUNT(CASE WHEN status = 'completed' OR status = 'mastered' THEN 1 END) as completed_lessons,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_lessons,
                    AVG(time_spent_minutes) as avg_time_per_lesson,
                    SUM(time_spent_minutes) as total_study_time,
                    MAX(phase) as current_phase,
                    MAX(week) as current_week
                FROM learning_progress
            `, (err, row) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(row);
            });
        });
    }

    formatDateTime(dateString) {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatDuration(minutes) {
        if (!minutes || minutes === 0) return 'N/A';
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        
        if (hours > 0) {
            return `${hours}h ${remainingMinutes}m`;
        }
        return `${remainingMinutes}m`;
    }

    generateMarkdownPortfolio(data) {
        const { profile, lessons, quests, skills, vault, analytics } = data;
        
        let markdown = '';
        
        // Header
        markdown += `# ${profile.username}'s Neural Odyssey Portfolio\n\n`;
        markdown += `> *A comprehensive machine learning journey from first principles to world-class practitioner*\n\n`;
        markdown += `**Generated:** ${new Date().toLocaleDateString()}\n\n`;
        
        // Learning Analytics Summary
        markdown += `## 📊 Learning Analytics\n\n`;
        markdown += `| Metric | Value |\n`;
        markdown += `|--------|-------|\n`;
        markdown += `| **Current Progress** | Phase ${analytics.current_phase}, Week ${analytics.current_week} |\n`;
        markdown += `| **Lessons Completed** | ${analytics.completed_lessons} |\n`;
        markdown += `| **Lessons Mastered** | ${analytics.mastered_lessons} |\n`;
        markdown += `| **Total Study Time** | ${this.formatDuration(analytics.total_study_time)} |\n`;
        markdown += `| **Average Time per Lesson** | ${this.formatDuration(Math.round(analytics.avg_time_per_lesson))} |\n`;
        markdown += `| **Current Streak** | ${profile.current_streak_days} days |\n`;
        markdown += `| **Longest Streak** | ${profile.longest_streak_days} days |\n\n`;

        // Skill Points
        if (skills.length > 0) {
            markdown += `## 🎯 Skill Mastery\n\n`;
            skills.forEach(skill => {
                const emoji = {
                    mathematics: '🔢',
                    programming: '💻',
                    theory: '📚',
                    applications: '🔧',
                    creativity: '🎨',
                    persistence: '💪'
                }[skill.category] || '⭐';
                
                markdown += `### ${emoji} ${skill.category.charAt(0).toUpperCase() + skill.category.slice(1)}\n`;
                markdown += `- **Total Points:** ${skill.total_points}\n`;
                markdown += `- **Achievements:** ${skill.achievements}\n\n`;
            });
        }

        // Vault Unlocks
        if (vault.length > 0) {
            markdown += `## 🗝️ Vault Discoveries\n\n`;
            markdown += `*Unlocked ${vault.length} mind-blowing secrets from the Neural Vault*\n\n`;
            
            const vaultByCategory = vault.reduce((acc, item) => {
                if (!acc[item.category]) acc[item.category] = [];
                acc[item.category].push(item);
                return acc;
            }, {});

            Object.entries(vaultByCategory).forEach(([category, items]) => {
                const categoryEmoji = {
                    secret_archives: '🗝️',
                    controversy_files: '⚔️',
                    beautiful_mind: '💎'
                }[category] || '📖';
                
                markdown += `### ${categoryEmoji} ${category.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}\n\n`;
                items.forEach(item => {
                    markdown += `- **${item.vault_item_id}** - Unlocked ${this.formatDateTime(item.unlocked_at)}\n`;
                    if (item.user_rating) {
                        markdown += `  - Rating: ${'⭐'.repeat(item.user_rating)}\n`;
                    }
                    if (item.user_notes) {
                        markdown += `  - Notes: *${item.user_notes}*\n`;
                    }
                });
                markdown += '\n';
            });
        }

        // Learning Progress by Phase
        if (lessons.length > 0) {
            markdown += `## 📚 Learning Journey\n\n`;
            
            const lessonsByPhase = lessons.reduce((acc, lesson) => {
                if (!acc[lesson.phase]) acc[acc.phase] = [];
                acc[lesson.phase].push(lesson);
                return acc;
            }, {});

            Object.entries(lessonsByPhase).forEach(([phase, phaseLessons]) => {
                const phaseNames = {
                    1: 'Mathematical Foundations and Historical Context',
                    2: 'Core Machine Learning with Deep Understanding', 
                    3: 'Advanced Topics and Modern AI',
                    4: 'Mastery and Innovation'
                };
                
                markdown += `### Phase ${phase}: ${phaseNames[phase] || 'Advanced Learning'}\n\n`;
                
                const lessonsByWeek = phaseLessons.reduce((acc, lesson) => {
                    if (!acc[lesson.week]) acc[lesson.week] = [];
                    acc[lesson.week].push(lesson);
                    return acc;
                }, {});

                Object.entries(lessonsByWeek).forEach(([week, weekLessons]) => {
                    markdown += `#### Week ${week}\n\n`;
                    weekLessons.forEach(lesson => {
                        const statusEmoji = lesson.status === 'mastered' ? '🏆' : '✅';
                        const typeEmoji = {
                            theory: '📖',
                            math: '🔢', 
                            visual: '📊',
                            coding: '💻'
                        }[lesson.lesson_type] || '📝';
                        
                        markdown += `- ${statusEmoji} ${typeEmoji} **${lesson.lesson_title}**\n`;
                        markdown += `  - Completed: ${this.formatDateTime(lesson.completed_at)}\n`;
                        markdown += `  - Time spent: ${this.formatDuration(lesson.time_spent_minutes)}\n`;
                        if (lesson.mastery_score) {
                            markdown += `  - Mastery: ${Math.round(lesson.mastery_score * 100)}%\n`;
                        }
                        if (lesson.notes) {
                            markdown += `  - Notes: *${lesson.notes}*\n`;
                        }
                        markdown += '\n';
                    });
                });
                markdown += '\n';
            });
        }

        // Completed Projects/Quests
        if (quests.length > 0) {
            markdown += `## 🚀 Projects & Achievements\n\n`;
            
            const questsByPhase = quests.reduce((acc, quest) => {
                if (!acc[quest.phase]) acc[quest.phase] = [];
                acc[quest.phase].push(quest);
                return acc;
            }, {});

            Object.entries(questsByPhase).forEach(([phase, phaseQuests]) => {
                markdown += `### Phase ${phase} Projects\n\n`;
                
                phaseQuests.forEach(quest => {
                    const difficultyStars = '⭐'.repeat(quest.difficulty_level);
                    const statusEmoji = quest.status === 'mastered' ? '🏆' : '✅';
                    const typeEmoji = {
                        coding_exercise: '💻',
                        implementation_project: '🔧',
                        theory_quiz: '📝',
                        practical_application: '🎯'
                    }[quest.quest_type] || '📋';
                    
                    markdown += `#### ${statusEmoji} ${typeEmoji} ${quest.quest_title}\n\n`;
                    markdown += `- **Difficulty:** ${difficultyStars} (${quest.difficulty_level}/5)\n`;
                    markdown += `- **Completed:** ${this.formatDateTime(quest.completed_at)}\n`;
                    markdown += `- **Time to Complete:** ${this.formatDuration(quest.time_to_complete_minutes)}\n`;
                    markdown += `- **Attempts:** ${quest.attempts_count}\n`;
                    
                    if (quest.hint_used_count > 0) {
                        markdown += `- **Hints Used:** ${quest.hint_used_count}\n`;
                    }
                    
                    if (quest.code_solution) {
                        markdown += `\n**Solution:**\n\`\`\`python\n${quest.code_solution}\n\`\`\`\n\n`;
                    }
                    
                    if (quest.execution_result) {
                        markdown += `**Output:**\n\`\`\`\n${quest.execution_result}\n\`\`\`\n\n`;
                    }
                    
                    if (quest.self_reflection) {
                        markdown += `**Reflection:** *${quest.self_reflection}*\n\n`;
                    }
                    
                    if (quest.mentor_feedback) {
                        markdown += `**AI Mentor Feedback:** ${quest.mentor_feedback}\n\n`;
                    }
                    
                    markdown += '---\n\n';
                });
            });
        }

        // Footer
        markdown += `## 🎯 Next Steps\n\n`;
        markdown += `Continue your Neural Odyssey journey at Phase ${analytics.current_phase}, Week ${analytics.current_week}.\n\n`;
        markdown += `*Generated by Neural Odyssey Portfolio Exporter*\n`;
        markdown += `*"The best way to understand AI is to build it from first principles"*\n`;

        return markdown;
    }

    generateJSONPortfolio(data) {
        return JSON.stringify({
            exportedAt: new Date().toISOString(),
            portfolioData: data,
            metadata: {
                exportOptions: options,
                version: '1.0.0',
                format: 'neural-odyssey-portfolio'
            }
        }, null, 2);
    }

    generateHTMLPortfolio(data) {
        const markdownContent = this.generateMarkdownPortfolio(data);
        
        // Simple HTML wrapper - in a real implementation you might use a markdown parser
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${data.profile.username}'s Neural Odyssey Portfolio</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 2rem; 
            line-height: 1.6;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        h1, h2, h3 { color: #00d4ff; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        th, td { border: 1px solid #333; padding: 0.5rem; text-align: left; }
        th { background: #1a1a1a; }
        code { background: #1a1a1a; padding: 0.2rem 0.4rem; border-radius: 3px; }
        pre { background: #1a1a1a; padding: 1rem; border-radius: 5px; overflow-x: auto; }
        .emoji { font-size: 1.2em; }
    </style>
</head>
<body>
    <pre>${markdownContent}</pre>
</body>
</html>`;
    }

    async exportPortfolio() {
        console.log(`📊 Exporting portfolio (Phase: ${options.phase}, Format: ${options.format})...`);
        
        // Gather all data
        const data = {
            profile: await this.getUserProfile(),
            lessons: await this.getCompletedLessons(options.phase, options.type),
            quests: await this.getCompletedQuests(options.phase),
            skills: await this.getSkillPoints(),
            vault: await this.getUnlockedVaultItems(),
            analytics: await getLearningAnalytics()
        };

        // Ensure output directory exists
        if (!fs.existsSync(OUTPUT_DIR)) {
            fs.mkdirSync(OUTPUT_DIR, { recursive: true });
            console.log(`✅ Created output directory: ${OUTPUT_DIR}`);
        }

        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
        const baseFilename = `neural-odyssey-portfolio-${timestamp}`;

        // Export based on format
        if (options.format === 'md' || options.format === 'all') {
            const markdown = this.generateMarkdownPortfolio(data);
            const mdPath = path.join(OUTPUT_DIR, `${baseFilename}.md`);
            fs.writeFileSync(mdPath, markdown);
            console.log(`✅ Exported Markdown: ${mdPath}`);
        }

        if (options.format === 'json' || options.format === 'all') {
            const json = this.generateJSONPortfolio(data);
            const jsonPath = path.join(OUTPUT_DIR, `${baseFilename}.json`);
            fs.writeFileSync(jsonPath, json);
            console.log(`✅ Exported JSON: ${jsonPath}`);
        }

        if (options.format === 'html' || options.format === 'all') {
            const html = this.generateHTMLPortfolio(data);
            const htmlPath = path.join(OUTPUT_DIR, `${baseFilename}.html`);
            fs.writeFileSync(htmlPath, html);
            console.log(`✅ Exported HTML: ${htmlPath}`);
        }

        // Summary
        console.log(`\n📈 Portfolio Summary:`);
        console.log(`   Lessons: ${data.lessons.length} completed`);
        console.log(`   Projects: ${data.quests.length} completed`);
        console.log(`   Vault Items: ${data.vault.length} unlocked`);
        console.log(`   Total Study Time: ${this.formatDuration(data.analytics.total_study_time)}`);
    }
}

async function main() {
    console.log('🎯 Neural Odyssey Portfolio Exporter\n');
    
    // Check if database exists
    if (!fs.existsSync(DB_PATH)) {
        console.error('❌ Database not found. Run "node scripts/init-db.js" first.');
        process.exit(1);
    }

    const exporter = new PortfolioExporter(DB_PATH);
    
    try {
        await exporter.connect();
        await exporter.exportPortfolio();
        await exporter.close();
        
        console.log('\n🎉 Portfolio export complete!');
        console.log(`📁 Check the ${options.output}/ directory for your files.`);
        
    } catch (error) {
        console.error('❌ Export failed:', error.message);
        process.exit(1);
    }
}

// Run the exporter
if (require.main === module) {
    main();
}

module.exports = PortfolioExporter;