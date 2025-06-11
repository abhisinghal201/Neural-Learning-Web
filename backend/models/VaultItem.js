/**
 * Neural Odyssey Vault Item Model
 * 
 * Represents vault reward items with unlock conditions and user interaction tracking.
 * Manages the three categories: Secret Archives, Controversy Files, Beautiful Mind Collection.
 * 
 * Author: Neural Explorer
 */

const db = require('../config/db');
const fs = require('fs');
const path = require('path');

class VaultItem {
    constructor(data = {}) {
        this.id = data.id || this.generateItemId();
        this.title = data.title || '';
        this.category = data.category || 'secret_archives'; // secret_archives, controversy_files, beautiful_mind
        this.icon = data.icon || '📖';
        this.unlock_condition = data.unlock_condition || data.unlockCondition || {};
        this.content = data.content || {};
        this.created_at = data.created_at || new Date().toISOString();
        
        // User interaction data
        this.unlocked = data.unlocked || false;
        this.unlocked_at = data.unlocked_at || null;
        this.is_read = data.is_read || false;
        this.read_at = data.read_at || null;
        this.user_rating = data.user_rating || null;
        this.user_notes = data.user_notes || null;
    }

    generateItemId() {
        const categoryPrefix = {
            'secret_archives': 'sa',
            'controversy_files': 'cf', 
            'beautiful_mind': 'bm'
        };
        const prefix = categoryPrefix[this.category] || 'vi';
        const timestamp = Date.now().toString().slice(-6);
        const random = Math.random().toString(36).substr(2, 3);
        return `${prefix}_${timestamp}_${random}`;
    }

    // ==========================================
    // VAULT ITEM LOADING AND MANAGEMENT
    // ==========================================

    static loadVaultItems() {
        try {
            const vaultPath = path.join(__dirname, '../../data/vault-items.json');
            if (!fs.existsSync(vaultPath)) {
                console.error('❌ Vault items file not found');
                return [];
            }

            const rawData = fs.readFileSync(vaultPath, 'utf8');
            const vaultData = JSON.parse(rawData);
            
            const allItems = [
                ...vaultData.secretArchives.map(item => new VaultItem({...item, category: 'secret_archives'})),
                ...vaultData.controversyFiles.map(item => new VaultItem({...item, category: 'controversy_files'})),
                ...vaultData.beautifulMindCollection.map(item => new VaultItem({...item, category: 'beautiful_mind'}))
            ];

            return allItems;
        } catch (error) {
            console.error('❌ Error loading vault items:', error);
            return [];
        }
    }

    static async getAllItems(includeContent = false) {
        try {
            const vaultItems = VaultItem.loadVaultItems();
            const unlockedItems = await db.query(`
                SELECT vault_item_id, unlocked_at, is_read, read_at, user_rating, user_notes
                FROM vault_unlocks
            `);
            
            const unlockedMap = new Map(unlockedItems.map(item => [item.vault_item_id, item]));
            
            return vaultItems.map(item => {
                const unlockData = unlockedMap.get(item.id);
                const vaultItem = new VaultItem({
                    ...item,
                    unlocked: !!unlockData,
                    unlocked_at: unlockData?.unlocked_at,
                    is_read: unlockData?.is_read || false,
                    read_at: unlockData?.read_at,
                    user_rating: unlockData?.user_rating,
                    user_notes: unlockData?.user_notes
                });

                // Hide content if not unlocked or not requested
                if (!vaultItem.unlocked || !includeContent) {
                    vaultItem.content = null;
                }

                return vaultItem;
            });
        } catch (error) {
            console.error('Error getting all vault items:', error);
            return [];
        }
    }

    static async getItemById(itemId, includeContent = true) {
        try {
            const allItems = await VaultItem.getAllItems(includeContent);
            return allItems.find(item => item.id === itemId) || null;
        } catch (error) {
            console.error('Error getting vault item by ID:', error);
            return null;
        }
    }

    static async getItemsByCategory(category, includeContent = false) {
        try {
            const allItems = await VaultItem.getAllItems(includeContent);
            return allItems.filter(item => item.category === category);
        } catch (error) {
            console.error('Error getting items by category:', error);
            return [];
        }
    }

    // ==========================================
    // UNLOCK CONDITION CHECKING
    // ==========================================

    static async checkUnlockConditions(phase, week, lessonId) {
        try {
            const vaultItems = VaultItem.loadVaultItems();
            const newUnlocks = [];

            for (const item of vaultItems) {
                // Check if already unlocked
                const existingUnlock = await db.get(`
                    SELECT * FROM vault_unlocks WHERE vault_item_id = ?
                `, [item.id]);

                if (existingUnlock) continue;

                // Check unlock condition
                if (VaultItem.evaluateUnlockCondition(item.unlock_condition, phase, week, lessonId)) {
                    await VaultItem.unlockItem(item.id, item.category);
                    newUnlocks.push({
                        id: item.id,
                        title: item.title,
                        category: item.category,
                        icon: item.icon
                    });
                }
            }

            return newUnlocks;
        } catch (error) {
            console.error('Error checking unlock conditions:', error);
            return [];
        }
    }

    static evaluateUnlockCondition(condition, currentPhase, currentWeek, currentLessonId) {
        switch (condition.type) {
            case 'lesson_complete':
                return (
                    currentPhase >= condition.phase &&
                    currentWeek >= condition.week &&
                    currentLessonId === condition.lesson
                );
            
            case 'phase_complete':
                return currentPhase > condition.phase;
            
            case 'week_complete':
                return (
                    currentPhase > condition.phase ||
                    (currentPhase === condition.phase && currentWeek > condition.week)
                );
            
            case 'multiple_lessons':
                return condition.lessons && condition.lessons.includes(currentLessonId);
            
            case 'progress_milestone':
                // Would check overall progress percentage
                return false; // Placeholder
            
            default:
                return false;
        }
    }

    static async unlockItem(itemId, category) {
        try {
            const result = await db.run(`
                INSERT INTO vault_unlocks (vault_item_id, category, unlocked_at)
                VALUES (?, ?, ?)
            `, [itemId, category, new Date().toISOString()]);

            console.log(`🎉 Unlocked vault item: ${itemId}`);
            return result;
        } catch (error) {
            console.error('Error unlocking vault item:', error);
            throw error;
        }
    }

    // ==========================================
    // USER INTERACTION METHODS
    // ==========================================

    async markAsRead() {
        try {
            const result = await db.run(`
                UPDATE vault_unlocks 
                SET is_read = 1, read_at = ?
                WHERE vault_item_id = ?
            `, [new Date().toISOString(), this.id]);

            this.is_read = true;
            this.read_at = new Date().toISOString();
            
            return result;
        } catch (error) {
            console.error('Error marking item as read:', error);
            throw error;
        }
    }

    async setRating(rating, notes = null) {
        try {
            if (rating < 1 || rating > 5) {
                throw new Error('Rating must be between 1 and 5');
            }

            const result = await db.run(`
                UPDATE vault_unlocks 
                SET user_rating = ?, user_notes = ?
                WHERE vault_item_id = ?
            `, [rating, notes, this.id]);

            this.user_rating = rating;
            this.user_notes = notes;
            
            return result;
        } catch (error) {
            console.error('Error setting rating:', error);
            throw error;
        }
    }

    async addNotes(notes) {
        try {
            const result = await db.run(`
                UPDATE vault_unlocks 
                SET user_notes = ?
                WHERE vault_item_id = ?
            `, [notes, this.id]);

            this.user_notes = notes;
            return result;
        } catch (error) {
            console.error('Error adding notes:', error);
            throw error;
        }
    }

    // ==========================================
    // VAULT ANALYTICS AND STATISTICS
    // ==========================================

    static async getUnlockStatistics() {
        try {
            const stats = await db.get(`
                SELECT 
                    COUNT(*) as total_unlocked,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as total_read,
                    AVG(user_rating) as average_rating,
                    MIN(unlocked_at) as first_unlock,
                    MAX(unlocked_at) as latest_unlock
                FROM vault_unlocks
            `);

            const categoryStats = await db.query(`
                SELECT 
                    category,
                    COUNT(*) as unlocked_count,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as read_count,
                    AVG(user_rating) as avg_rating
                FROM vault_unlocks
                GROUP BY category
            `);

            const totalItems = VaultItem.loadVaultItems().length;

            return {
                overview: {
                    ...stats,
                    total_available: totalItems,
                    unlock_percentage: totalItems > 0 ? ((stats.total_unlocked / totalItems) * 100).toFixed(1) : 0
                },
                by_category: categoryStats
            };
        } catch (error) {
            console.error('Error getting unlock statistics:', error);
            return null;
        }
    }

    static async getReadingEngagement() {
        try {
            const engagement = await db.query(`
                SELECT 
                    vault_item_id,
                    category,
                    unlocked_at,
                    is_read,
                    read_at,
                    user_rating,
                    CASE 
                        WHEN read_at IS NOT NULL AND unlocked_at IS NOT NULL 
                        THEN (julianday(read_at) - julianday(unlocked_at)) * 24 * 60
                        ELSE NULL 
                    END as minutes_to_read
                FROM vault_unlocks
                ORDER BY unlocked_at DESC
            `);

            const avgTimeToRead = engagement
                .filter(item => item.minutes_to_read !== null)
                .reduce((sum, item) => sum + item.minutes_to_read, 0) / 
                engagement.filter(item => item.minutes_to_read !== null).length;

            return {
                items: engagement,
                average_time_to_read_minutes: avgTimeToRead || 0,
                read_rate: engagement.length > 0 ? 
                    (engagement.filter(item => item.is_read).length / engagement.length * 100).toFixed(1) : 0
            };
        } catch (error) {
            console.error('Error getting reading engagement:', error);
            return null;
        }
    }

    static async getUnlockTimeline() {
        try {
            const timeline = await db.query(`
                SELECT 
                    vault_item_id,
                    category,
                    unlocked_at,
                    is_read,
                    user_rating
                FROM vault_unlocks
                ORDER BY unlocked_at ASC
            `);

            // Enrich with vault item details
            const vaultItems = VaultItem.loadVaultItems();
            const itemMap = new Map(vaultItems.map(item => [item.id, item]));

            const enrichedTimeline = timeline.map(unlock => {
                const vaultItem = itemMap.get(unlock.vault_item_id);
                return {
                    ...unlock,
                    title: vaultItem?.title || 'Unknown Item',
                    icon: vaultItem?.icon || '📖'
                };
            });

            return enrichedTimeline;
        } catch (error) {
            console.error('Error getting unlock timeline:', error);
            return [];
        }
    }

    // ==========================================
    // SEARCH AND FILTERING
    // ==========================================

    static async searchItems(query, filters = {}) {
        try {
            const allItems = await VaultItem.getAllItems(false);
            let filteredItems = allItems;

            // Apply category filter
            if (filters.category) {
                filteredItems = filteredItems.filter(item => item.category === filters.category);
            }

            // Apply unlock status filter
            if (filters.unlocked !== undefined) {
                filteredItems = filteredItems.filter(item => item.unlocked === filters.unlocked);
            }

            // Apply read status filter
            if (filters.read !== undefined) {
                filteredItems = filteredItems.filter(item => item.is_read === filters.read);
            }

            // Apply rating filter
            if (filters.min_rating) {
                filteredItems = filteredItems.filter(item => 
                    item.user_rating && item.user_rating >= filters.min_rating
                );
            }

            // Apply text search
            if (query && query.trim()) {
                const searchTerm = query.toLowerCase().trim();
                filteredItems = filteredItems.filter(item =>
                    item.title.toLowerCase().includes(searchTerm) ||
                    (item.content && item.content.headline && 
                     item.content.headline.toLowerCase().includes(searchTerm))
                );
            }

            return filteredItems;
        } catch (error) {
            console.error('Error searching vault items:', error);
            return [];
        }
    }

    static async getUpcomingUnlocks(currentPhase, currentWeek) {
        try {
            const vaultItems = VaultItem.loadVaultItems();
            const unlockedIds = new Set((await db.query(`
                SELECT vault_item_id FROM vault_unlocks
            `)).map(u => u.vault_item_id));

            const upcomingItems = vaultItems
                .filter(item => !unlockedIds.has(item.id))
                .filter(item => {
                    const condition = item.unlock_condition;
                    if (condition.type === 'lesson_complete') {
                        return condition.phase <= currentPhase + 1;
                    }
                    return true;
                })
                .slice(0, 5)
                .map(item => ({
                    id: item.id,
                    title: item.title,
                    category: item.category,
                    icon: item.icon,
                    unlock_condition: item.unlock_condition,
                    estimated_unlock: VaultItem.estimateUnlockTime(item.unlock_condition, currentPhase, currentWeek)
                }));

            return upcomingItems;
        } catch (error) {
            console.error('Error getting upcoming unlocks:', error);
            return [];
        }
    }

    static estimateUnlockTime(condition, currentPhase, currentWeek) {
        if (condition.type === 'lesson_complete') {
            const weeksAway = (condition.phase - currentPhase) * 12 + (condition.week - currentWeek);
            
            if (weeksAway <= 0) return 'Available soon';
            if (weeksAway === 1) return 'Next week';
            if (weeksAway <= 4) return `${weeksAway} weeks`;
            return `Phase ${condition.phase}`;
        }
        
        return 'Progress dependent';
    }

    // ==========================================
    // UTILITY METHODS
    // ==========================================

    isUnlocked() {
        return this.unlocked;
    }

    canViewContent() {
        return this.unlocked;
    }

    getPublicData() {
        // Return safe data for unlocked items
        if (!this.unlocked) {
            return {
                id: this.id,
                title: this.title,
                category: this.category,
                icon: this.icon,
                unlocked: false,
                unlock_condition: this.unlock_condition
            };
        }

        return {
            id: this.id,
            title: this.title,
            category: this.category,
            icon: this.icon,
            unlocked: true,
            unlocked_at: this.unlocked_at,
            is_read: this.is_read,
            read_at: this.read_at,
            user_rating: this.user_rating,
            user_notes: this.user_notes,
            content: this.content
        };
    }

    toJSON() {
        return this.getPublicData();
    }

    static validateUnlockCondition(condition) {
        const validTypes = ['lesson_complete', 'phase_complete', 'week_complete', 'multiple_lessons', 'progress_milestone'];
        
        if (!condition.type || !validTypes.includes(condition.type)) {
            return false;
        }

        switch (condition.type) {
            case 'lesson_complete':
                return condition.phase && condition.week && condition.lesson;
            case 'phase_complete':
                return condition.phase;
            case 'week_complete':
                return condition.phase && condition.week;
            case 'multiple_lessons':
                return Array.isArray(condition.lessons) && condition.lessons.length > 0;
            case 'progress_milestone':
                return condition.milestone_type && condition.threshold;
            default:
                return false;
        }
    }

    static getCategoryEmoji(category) {
        const emojiMap = {
            'secret_archives': '🗝️',
            'controversy_files': '⚔️',
            'beautiful_mind': '💎'
        };
        return emojiMap[category] || '📖';
    }

    static getCategoryName(category) {
        const nameMap = {
            'secret_archives': 'Secret Archives',
            'controversy_files': 'Controversy Files',
            'beautiful_mind': 'Beautiful Mind Collection'
        };
        return nameMap[category] || 'Unknown Category';
    }
}

module.exports = VaultItem;