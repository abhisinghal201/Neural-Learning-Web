/**
 * Neural Odyssey Vault Controller
 * 
 * Manages the Neural Vault reward system including:
 * - 🗝️ Secret Archives: Mind-blowing historical connections
 * - ⚔️ Controversy Files: Drama and feuds in AI history  
 * - 💎 Beautiful Mind Collection: Mathematical elegance
 * 
 * Features:
 * - Automatic unlock checking based on lesson completion
 * - Vault item serving with spoiler protection
 * - User interaction tracking (ratings, notes, read status)
 * - Unlock analytics and progress tracking
 * 
 * Connects to: backend/routes/vault.js + backend/config/db.js
 * Uses: data/vault-items.json + vault_unlocks table
 * 
 * Author: Neural Explorer
 */

const db = require('../config/db');
const fs = require('fs');
const path = require('path');
const { validationResult } = require('express-validator');

// Load vault items from JSON file
const VAULT_ITEMS_PATH = path.join(__dirname, '../../data/vault-items.json');

class VaultController {
    
    constructor() {
        this.vaultItems = null;
        this.loadVaultItems();
    }
    
    // ==========================================
    // VAULT ITEMS MANAGEMENT
    // ==========================================
    
    /**
     * Load vault items from JSON file
     */
    loadVaultItems() {
        try {
            if (fs.existsSync(VAULT_ITEMS_PATH)) {
                const rawData = fs.readFileSync(VAULT_ITEMS_PATH, 'utf8');
                this.vaultItems = JSON.parse(rawData);
                console.log('✅ Vault items loaded successfully');
            } else {
                console.error('❌ Vault items file not found:', VAULT_ITEMS_PATH);
                this.vaultItems = { secretArchives: [], controversyFiles: [], beautifulMindCollection: [] };
            }
        } catch (error) {
            console.error('❌ Failed to load vault items:', error.message);
            this.vaultItems = { secretArchives: [], controversyFiles: [], beautifulMindCollection: [] };
        }
    }
    
    /**
     * Get all vault items with unlock status
     * GET /api/v1/vault/items
     */
    async getAllVaultItems(req, res, next) {
        try {
            const { category, unlocked_only = false } = req.query;
            
            // Get all unlocked vault items from database
            const unlockedItems = await db.query(`
                SELECT vault_item_id, category, unlocked_at, is_read, read_at, user_rating, user_notes
                FROM vault_unlocks
                ORDER BY unlocked_at DESC
            `);
            
            const unlockedIds = new Set(unlockedItems.map(item => item.vault_item_id));
            
            // Combine all vault categories
            const allItems = [
                ...this.vaultItems.secretArchives,
                ...this.vaultItems.controversyFiles,
                ...this.vaultItems.beautifulMindCollection
            ];
            
            // Process items with unlock status
            let processedItems = allItems.map(item => {
                const isUnlocked = unlockedIds.has(item.id);
                const unlockData = unlockedItems.find(u => u.vault_item_id === item.id);
                
                if (isUnlocked) {
                    // Return full content for unlocked items
                    return {
                        ...item,
                        unlocked: true,
                        unlocked_at: unlockData?.unlocked_at,
                        is_read: unlockData?.is_read || false,
                        read_at: unlockData?.read_at,
                        user_rating: unlockData?.user_rating,
                        user_notes: unlockData?.user_notes
                    };
                } else {
                    // Return minimal info for locked items (spoiler protection)
                    return {
                        id: item.id,
                        title: item.title,
                        category: item.category,
                        icon: item.icon,
                        unlocked: false,
                        unlock_condition: item.unlockCondition,
                        content: null // Hidden until unlocked
                    };
                }
            });
            
            // Filter by category if specified
            if (category) {
                processedItems = processedItems.filter(item => item.category === category);
            }
            
            // Filter by unlocked status if specified
            if (unlocked_only === 'true') {
                processedItems = processedItems.filter(item => item.unlocked);
            }
            
            // Group by category for better organization
            const groupedItems = {
                secret_archives: processedItems.filter(item => item.category === 'secret_archives'),
                controversy_files: processedItems.filter(item => item.category === 'controversy_files'),
                beautiful_mind: processedItems.filter(item => item.category === 'beautiful_mind')
            };
            
            // Calculate statistics
            const stats = {
                total_items: allItems.length,
                unlocked_items: unlockedItems.length,
                unlock_percentage: ((unlockedItems.length / allItems.length) * 100).toFixed(1),
                by_category: {
                    secret_archives: {
                        total: this.vaultItems.secretArchives.length,
                        unlocked: unlockedItems.filter(u => u.category === 'secret_archives').length
                    },
                    controversy_files: {
                        total: this.vaultItems.controversyFiles.length,
                        unlocked: unlockedItems.filter(u => u.category === 'controversy_files').length
                    },
                    beautiful_mind: {
                        total: this.vaultItems.beautifulMindCollection.length,
                        unlocked: unlockedItems.filter(u => u.category === 'beautiful_mind').length
                    }
                }
            };
            
            res.json({
                success: true,
                data: {
                    items: groupedItems,
                    statistics: stats
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    /**
     * Get a specific vault item by ID
     * GET /api/v1/vault/items/:itemId
     */
    async getVaultItem(req, res, next) {
        try {
            const { itemId } = req.params;
            
            // Check if item is unlocked
            const unlockData = await db.get(`
                SELECT * FROM vault_unlocks WHERE vault_item_id = ?
            `, [itemId]);
            
            if (!unlockData) {
                return res.status(403).json({
                    success: false,
                    message: 'Vault item not unlocked yet',
                    hint: 'Complete more lessons to unlock this reward!'
                });
            }
            
            // Find the item in vault data
            const allItems = [
                ...this.vaultItems.secretArchives,
                ...this.vaultItems.controversyFiles,
                ...this.vaultItems.beautifulMindCollection
            ];
            
            const vaultItem = allItems.find(item => item.id === itemId);
            
            if (!vaultItem) {
                return res.status(404).json({
                    success: false,
                    message: 'Vault item not found'
                });
            }
            
            // Return full item with unlock data
            res.json({
                success: true,
                data: {
                    ...vaultItem,
                    unlocked: true,
                    unlocked_at: unlockData.unlocked_at,
                    is_read: unlockData.is_read,
                    read_at: unlockData.read_at,
                    user_rating: unlockData.user_rating,
                    user_notes: unlockData.user_notes
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    /**
     * Mark vault item as read
     * POST /api/v1/vault/items/:itemId/read
     */
    async markAsRead(req, res, next) {
        try {
            const { itemId } = req.params;
            
            // Check if item is unlocked
            const unlockData = await db.get(`
                SELECT * FROM vault_unlocks WHERE vault_item_id = ?
            `, [itemId]);
            
            if (!unlockData) {
                return res.status(403).json({
                    success: false,
                    message: 'Vault item not unlocked'
                });
            }
            
            // Mark as read
            await db.run(`
                UPDATE vault_unlocks 
                SET is_read = 1, read_at = ?
                WHERE vault_item_id = ?
            `, [new Date().toISOString(), itemId]);
            
            res.json({
                success: true,
                message: 'Vault item marked as read',
                data: {
                    item_id: itemId,
                    read_at: new Date().toISOString()
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    /**
     * Rate a vault item
     * POST /api/v1/vault/items/:itemId/rate
     */
    async rateVaultItem(req, res, next) {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }
            
            const { itemId } = req.params;
            const { rating, notes } = req.body;
            
            if (rating < 1 || rating > 5) {
                return res.status(400).json({
                    success: false,
                    message: 'Rating must be between 1 and 5'
                });
            }
            
            // Check if item is unlocked
            const unlockData = await db.get(`
                SELECT * FROM vault_unlocks WHERE vault_item_id = ?
            `, [itemId]);
            
            if (!unlockData) {
                return res.status(403).json({
                    success: false,
                    message: 'Vault item not unlocked'
                });
            }
            
            // Update rating and notes
            await db.run(`
                UPDATE vault_unlocks 
                SET user_rating = ?, user_notes = ?
                WHERE vault_item_id = ?
            `, [rating, notes || null, itemId]);
            
            res.json({
                success: true,
                message: 'Rating saved successfully',
                data: {
                    item_id: itemId,
                    rating,
                    notes
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    // ==========================================
    // UNLOCK SYSTEM
    // ==========================================
    
    /**
     * Check and process vault unlocks based on lesson completion
     * Called internally by learning controller
     */
    async checkUnlocks(phase, week, lessonId) {
        try {
            console.log(`🗝️ Checking vault unlocks for Phase ${phase}, Week ${week}, Lesson ${lessonId}`);
            
            // Get all vault items
            const allItems = [
                ...this.vaultItems.secretArchives,
                ...this.vaultItems.controversyFiles,
                ...this.vaultItems.beautifulMindCollection
            ];
            
            const newUnlocks = [];
            
            for (const item of allItems) {
                // Check if already unlocked
                const existingUnlock = await db.get(`
                    SELECT * FROM vault_unlocks WHERE vault_item_id = ?
                `, [item.id]);
                
                if (existingUnlock) {
                    continue; // Already unlocked
                }
                
                // Check unlock condition
                if (this.checkUnlockCondition(item.unlockCondition, phase, week, lessonId)) {
                    // Unlock the item!
                    await db.run(`
                        INSERT INTO vault_unlocks (vault_item_id, category, unlocked_at)
                        VALUES (?, ?, ?)
                    `, [item.id, item.category, new Date().toISOString()]);
                    
                    newUnlocks.push({
                        id: item.id,
                        title: item.title,
                        category: item.category,
                        icon: item.icon
                    });
                    
                    console.log(`🎉 Unlocked vault item: ${item.title}`);
                }
            }
            
            return newUnlocks;
            
        } catch (error) {
            console.error('❌ Error checking vault unlocks:', error);
            return [];
        }
    }
    
    /**
     * Check if unlock condition is met
     */
    checkUnlockCondition(condition, currentPhase, currentWeek, currentLessonId) {
        if (condition.type === 'lesson_complete') {
            return (
                currentPhase >= condition.phase &&
                currentWeek >= condition.week &&
                currentLessonId === condition.lesson
            );
        }
        
        // Add other condition types as needed
        return false;
    }
    
    /**
     * Manually trigger unlock check (for testing/admin)
     * POST /api/v1/vault/check-unlocks
     */
    async triggerUnlockCheck(req, res, next) {
        try {
            const { phase, week, lesson } = req.body;
            
            if (!phase || !week || !lesson) {
                return res.status(400).json({
                    success: false,
                    message: 'Phase, week, and lesson are required'
                });
            }
            
            const newUnlocks = await this.checkUnlocks(phase, week, lesson);
            
            res.json({
                success: true,
                message: `Checked unlocks for Phase ${phase}, Week ${week}, Lesson ${lesson}`,
                data: {
                    new_unlocks: newUnlocks,
                    unlock_count: newUnlocks.length
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    // ==========================================
    // VAULT ANALYTICS
    // ==========================================
    
    /**
     * Get vault statistics and analytics
     * GET /api/v1/vault/analytics
     */
    async getVaultAnalytics(req, res, next) {
        try {
            // Overall statistics
            const overallStats = await db.get(`
                SELECT 
                    COUNT(*) as total_unlocked,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as total_read,
                    AVG(user_rating) as average_rating,
                    MIN(unlocked_at) as first_unlock,
                    MAX(unlocked_at) as latest_unlock
                FROM vault_unlocks
            `);
            
            // Category breakdown
            const categoryStats = await db.query(`
                SELECT 
                    category,
                    COUNT(*) as unlocked_count,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as read_count,
                    AVG(user_rating) as avg_rating
                FROM vault_unlocks
                GROUP BY category
                ORDER BY unlocked_count DESC
            `);
            
            // Recent unlocks
            const recentUnlocks = await db.query(`
                SELECT vault_item_id, category, unlocked_at
                FROM vault_unlocks
                ORDER BY unlocked_at DESC
                LIMIT 5
            `);
            
            // Reading engagement
            const engagementStats = await db.get(`
                SELECT 
                    COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as highly_rated,
                    COUNT(CASE WHEN user_notes IS NOT NULL AND user_notes != '' THEN 1 END) as with_notes,
                    COUNT(CASE WHEN is_read = 1 THEN 1 END) as read_items
                FROM vault_unlocks
            `);
            
            // Calculate total available items
            const totalItems = this.vaultItems.secretArchives.length + 
                             this.vaultItems.controversyFiles.length + 
                             this.vaultItems.beautifulMindCollection.length;
            
            res.json({
                success: true,
                data: {
                    overview: {
                        ...overallStats,
                        total_available: totalItems,
                        unlock_percentage: totalItems > 0 ? ((overallStats.total_unlocked / totalItems) * 100).toFixed(1) : 0,
                        read_percentage: overallStats.total_unlocked > 0 ? ((overallStats.total_read / overallStats.total_unlocked) * 100).toFixed(1) : 0
                    },
                    by_category: categoryStats,
                    recent_unlocks: recentUnlocks,
                    engagement: engagementStats,
                    generated_at: new Date().toISOString()
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    /**
     * Get vault unlock timeline
     * GET /api/v1/vault/timeline
     */
    async getUnlockTimeline(req, res, next) {
        try {
            const timeline = await db.query(`
                SELECT 
                    vault_item_id,
                    category,
                    unlocked_at,
                    is_read,
                    read_at,
                    user_rating
                FROM vault_unlocks
                ORDER BY unlocked_at ASC
            `);
            
            // Add vault item details
            const allItems = [
                ...this.vaultItems.secretArchives,
                ...this.vaultItems.controversyFiles,
                ...this.vaultItems.beautifulMindCollection
            ];
            
            const enrichedTimeline = timeline.map(unlock => {
                const vaultItem = allItems.find(item => item.id === unlock.vault_item_id);
                return {
                    ...unlock,
                    title: vaultItem?.title || 'Unknown Item',
                    icon: vaultItem?.icon || '📖'
                };
            });
            
            res.json({
                success: true,
                data: {
                    timeline: enrichedTimeline,
                    total_unlocks: timeline.length
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    // ==========================================
    // UTILITY METHODS
    // ==========================================
    
    /**
     * Get upcoming vault rewards (what can be unlocked next)
     * GET /api/v1/vault/upcoming
     */
    async getUpcomingRewards(req, res, next) {
        try {
            // Get current progress
            const currentProgress = await db.get(`
                SELECT current_phase, current_week FROM user_profile WHERE id = 1
            `);
            
            if (!currentProgress) {
                return res.status(404).json({
                    success: false,
                    message: 'User profile not found'
                });
            }
            
            // Get unlocked item IDs
            const unlockedIds = await db.query(`
                SELECT vault_item_id FROM vault_unlocks
            `);
            const unlockedSet = new Set(unlockedIds.map(u => u.vault_item_id));
            
            // Find upcoming rewards
            const allItems = [
                ...this.vaultItems.secretArchives,
                ...this.vaultItems.controversyFiles,
                ...this.vaultItems.beautifulMindCollection
            ];
            
            const upcomingRewards = allItems
                .filter(item => !unlockedSet.has(item.id))
                .filter(item => {
                    const condition = item.unlockCondition;
                    if (condition.type === 'lesson_complete') {
                        // Show items that are close to being unlocked
                        return condition.phase <= currentProgress.current_phase + 1;
                    }
                    return true;
                })
                .slice(0, 5) // Limit to next 5 rewards
                .map(item => ({
                    id: item.id,
                    title: item.title,
                    category: item.category,
                    icon: item.icon,
                    unlock_condition: item.unlockCondition,
                    estimated_unlock: this.estimateUnlockTime(item.unlockCondition, currentProgress)
                }));
            
            res.json({
                success: true,
                data: {
                    upcoming_rewards: upcomingRewards,
                    current_progress: currentProgress
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
    
    /**
     * Estimate when a vault item might be unlocked
     */
    estimateUnlockTime(condition, currentProgress) {
        if (condition.type === 'lesson_complete') {
            const weeksAway = (condition.phase - currentProgress.current_phase) * 12 + 
                            (condition.week - currentProgress.current_week);
            
            if (weeksAway <= 0) {
                return 'Available soon';
            } else if (weeksAway === 1) {
                return 'Next week';
            } else if (weeksAway <= 4) {
                return `${weeksAway} weeks`;
            } else {
                return `Phase ${condition.phase}`;
            }
        }
        
        return 'Progress dependent';
    }
    
    /**
     * Reload vault items from file (for development)
     * POST /api/v1/vault/reload
     */
    async reloadVaultItems(req, res, next) {
        try {
            this.loadVaultItems();
            
            res.json({
                success: true,
                message: 'Vault items reloaded successfully',
                data: {
                    secret_archives: this.vaultItems.secretArchives.length,
                    controversy_files: this.vaultItems.controversyFiles.length,
                    beautiful_mind: this.vaultItems.beautifulMindCollection.length
                }
            });
            
        } catch (error) {
            next(error);
        }
    }
}

module.exports = new VaultController();