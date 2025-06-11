/**
 * Neural Odyssey Vault Routes
 * 
 * API endpoints for managing vault items - the reward system that unlocks
 * mind-blowing secrets, controversies, and beautiful mathematical insights
 * as the user progresses through their learning journey.
 * 
 * Vault Categories:
 * 🗝️ Secret Archives - Historical mind-blowers
 * ⚔️ Controversy Files - Academic drama and feuds  
 * 💎 Beautiful Mind Collection - Mathematical elegance
 * 
 * Author: Neural Explorer
 */

const express = require('express');
const { body, param, query, validationResult } = require('express-validator');
const fs = require('fs');
const path = require('path');
const db = require('../config/db');
const vaultController = require('../controllers/vaultController');

const router = express.Router();

// Load vault items from JSON file
const VAULT_ITEMS_PATH = path.join(__dirname, '../../data/vault-items.json');
let vaultItemsData = {};

try {
    if (fs.existsSync(VAULT_ITEMS_PATH)) {
        vaultItemsData = JSON.parse(fs.readFileSync(VAULT_ITEMS_PATH, 'utf8'));
        console.log('✅ Vault items loaded successfully');
    } else {
        console.warn('⚠️ Vault items file not found');
    }
} catch (error) {
    console.error('❌ Failed to load vault items:', error.message);
}

// Validation middleware
const handleValidationErrors = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            success: false,
            error: 'Validation failed',
            details: errors.array()
        });
    }
    next();
};

/**
 * GET /api/v1/vault/items
 * Get all unlocked vault items for the user
 */
router.get('/items', async (req, res, next) => {
    try {
        const unlockedItems = await db.query(`
            SELECT 
                vu.*,
                datetime(vu.unlocked_at, 'localtime') as unlocked_at_local
            FROM vault_unlocks vu
            ORDER BY vu.unlocked_at DESC
        `);

        // Enrich with content from vault-items.json
        const enrichedItems = unlockedItems.map(item => {
            let content = null;
            
            // Find the content in the JSON data
            const allItems = [
                ...vaultItemsData.secretArchives || [],
                ...vaultItemsData.controversyFiles || [],
                ...vaultItemsData.beautifulMindCollection || []
            ];
            
            const vaultItem = allItems.find(vi => vi.id === item.vault_item_id);
            if (vaultItem) {
                content = vaultItem.content;
            }

            return {
                ...item,
                content,
                category_display: item.category.split('_').map(w => 
                    w.charAt(0).toUpperCase() + w.slice(1)
                ).join(' ')
            };
        });

        res.json({
            success: true,
            data: {
                items: enrichedItems,
                total: enrichedItems.length,
                categories: {
                    secret_archives: enrichedItems.filter(i => i.category === 'secret_archives').length,
                    controversy_files: enrichedItems.filter(i => i.category === 'controversy_files').length,
                    beautiful_mind: enrichedItems.filter(i => i.category === 'beautiful_mind').length
                }
            }
        });

    } catch (error) {
        next(error);
    }
});

/**
 * GET /api/v1/vault/items/:itemId
 * Get specific vault item details
 */
router.get('/items/:itemId', 
    [
        param('itemId').notEmpty().withMessage('Item ID is required')
    ],
    handleValidationErrors,
    async (req, res, next) => {
        try {
            const { itemId } = req.params;

            // Check if item is unlocked
            const unlockedItem = await db.get(`
                SELECT 
                    vu.*,
                    datetime(vu.unlocked_at, 'localtime') as unlocked_at_local,
                    datetime(vu.read_at, 'localtime') as read_at_local
                FROM vault_unlocks vu
                WHERE vu.vault_item_id = ?
            `, [itemId]);

            if (!unlockedItem) {
                return res.status(404).json({
                    success: false,
                    error: 'Vault item not found or not unlocked'
                });
            }

            // Get content from JSON
            const allItems = [
                ...vaultItemsData.secretArchives || [],
                ...vaultItemsData.controversyFiles || [],
                ...vaultItemsData.beautifulMindCollection || []
            ];
            
            const vaultItem = allItems.find(vi => vi.id === itemId);
            if (!vaultItem) {
                return res.status(404).json({
                    success: false,
                    error: 'Vault item content not found'
                });
            }

            // Mark as read if not already read
            if (!unlockedItem.is_read) {
                await db.run(`
                    UPDATE vault_unlocks 
                    SET is_read = 1, read_at = CURRENT_TIMESTAMP
                    WHERE vault_item_id = ?
                `, [itemId]);
            }

            res.json({
                success: true,
                data: {
                    ...unlockedItem,
                    content: vaultItem.content,
                    title: vaultItem.title,
                    icon: vaultItem.icon,
                    category_display: unlockedItem.category.split('_').map(w => 
                        w.charAt(0).toUpperCase() + w.slice(1)
                    ).join(' ')
                }
            });

        } catch (error) {
            next(error);
        }
    }
);

/**
 * POST /api/v1/vault/items/:itemId/rate
 * Rate a vault item (1-5 stars)
 */
router.post('/items/:itemId/rate',
    [
        param('itemId').notEmpty().withMessage('Item ID is required'),
        body('rating').isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1 and 5'),
        body('notes').optional().isString().isLength({ max: 500 }).withMessage('Notes must be under 500 characters')
    ],
    handleValidationErrors,
    async (req, res, next) => {
        try {
            const { itemId } = req.params;
            const { rating, notes } = req.body;

            // Check if item is unlocked
            const unlockedItem = await db.get(`
                SELECT vault_item_id FROM vault_unlocks WHERE vault_item_id = ?
            `, [itemId]);

            if (!unlockedItem) {
                return res.status(404).json({
                    success: false,
                    error: 'Vault item not found or not unlocked'
                });
            }

            // Update rating
            const result = await db.run(`
                UPDATE vault_unlocks 
                SET user_rating = ?, user_notes = ?
                WHERE vault_item_id = ?
            `, [rating, notes || null, itemId]);

            if (result.changes === 0) {
                return res.status(404).json({
                    success: false,
                    error: 'Failed to update rating'
                });
            }

            res.json({
                success: true,
                message: 'Rating updated successfully',
                data: {
                    itemId,
                    rating,
                    notes
                }
            });

        } catch (error) {
            next(error);
        }
    }
);

/**
 * GET /api/v1/vault/check-unlocks
 * Check for new vault items that should be unlocked based on current progress
 */
router.get('/check-unlocks', async (req, res, next) => {
    try {
        const newUnlocks = await vaultController.checkAndUnlockItems();
        
        res.json({
            success: true,
            data: {
                newUnlocks,
                message: newUnlocks.length > 0 
                    ? `🎉 ${newUnlocks.length} new vault item(s) unlocked!`
                    : 'No new unlocks available'
            }
        });

    } catch (error) {
        next(error);
    }
});

/**
 * GET /api/v1/vault/stats
 * Get vault statistics and progress
 */
router.get('/stats', async (req, res, next) => {
    try {
        const stats = await db.get(`
            SELECT 
                COUNT(*) as total_unlocked,
                COUNT(CASE WHEN is_read = 1 THEN 1 END) as total_read,
                COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) as total_rated,
                AVG(user_rating) as average_rating,
                COUNT(CASE WHEN category = 'secret_archives' THEN 1 END) as secret_archives_count,
                COUNT(CASE WHEN category = 'controversy_files' THEN 1 END) as controversy_files_count,
                COUNT(CASE WHEN category = 'beautiful_mind' THEN 1 END) as beautiful_mind_count
            FROM vault_unlocks
        `);

        // Calculate total available items
        const totalAvailable = (vaultItemsData.secretArchives?.length || 0) +
                             (vaultItemsData.controversyFiles?.length || 0) +
                             (vaultItemsData.beautifulMindCollection?.length || 0);

        // Get recent unlocks
        const recentUnlocks = await db.query(`
            SELECT 
                vault_item_id,
                category,
                unlocked_at,
                is_read
            FROM vault_unlocks
            ORDER BY unlocked_at DESC
            LIMIT 5
        `);

        res.json({
            success: true,
            data: {
                overview: {
                    totalAvailable,
                    totalUnlocked: stats.total_unlocked || 0,
                    totalRead: stats.total_read || 0,
                    totalRated: stats.total_rated || 0,
                    averageRating: stats.average_rating ? Math.round(stats.average_rating * 10) / 10 : null,
                    completionPercentage: totalAvailable > 0 
                        ? Math.round((stats.total_unlocked / totalAvailable) * 100)
                        : 0
                },
                categories: {
                    secretArchives: {
                        available: vaultItemsData.secretArchives?.length || 0,
                        unlocked: stats.secret_archives_count || 0
                    },
                    controversyFiles: {
                        available: vaultItemsData.controversyFiles?.length || 0,
                        unlocked: stats.controversy_files_count || 0
                    },
                    beautifulMind: {
                        available: vaultItemsData.beautifulMindCollection?.length || 0,
                        unlocked: stats.beautiful_mind_count || 0
                    }
                },
                recentActivity: recentUnlocks.map(item => ({
                    ...item,
                    unlocked_at_relative: getRelativeTime(item.unlocked_at)
                }))
            }
        });

    } catch (error) {
        next(error);
    }
});

/**
 * GET /api/v1/vault/categories/:category
 * Get vault items by category
 */
router.get('/categories/:category',
    [
        param('category').isIn(['secret_archives', 'controversy_files', 'beautiful_mind'])
            .withMessage('Invalid category')
    ],
    handleValidationErrors,
    async (req, res, next) => {
        try {
            const { category } = req.params;

            const items = await db.query(`
                SELECT 
                    vu.*,
                    datetime(vu.unlocked_at, 'localtime') as unlocked_at_local
                FROM vault_unlocks vu
                WHERE vu.category = ?
                ORDER BY vu.unlocked_at DESC
            `, [category]);

            // Enrich with content
            const enrichedItems = items.map(item => {
                const categoryData = {
                    secret_archives: vaultItemsData.secretArchives || [],
                    controversy_files: vaultItemsData.controversyFiles || [],
                    beautiful_mind: vaultItemsData.beautifulMindCollection || []
                }[category];

                const vaultItem = categoryData.find(vi => vi.id === item.vault_item_id);
                
                return {
                    ...item,
                    title: vaultItem?.title || 'Unknown',
                    icon: vaultItem?.icon || '📖',
                    preview: vaultItem?.content?.headline || ''
                };
            });

            const categoryDisplayName = category.split('_').map(w => 
                w.charAt(0).toUpperCase() + w.slice(1)
            ).join(' ');

            res.json({
                success: true,
                data: {
                    category: categoryDisplayName,
                    items: enrichedItems,
                    total: enrichedItems.length
                }
            });

        } catch (error) {
            next(error);
        }
    }
);

/**
 * Utility function to get relative time
 */
function getRelativeTime(dateString) {
    const now = new Date();
    const past = new Date(dateString);
    const diffMs = now - past;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffDays > 0) {
        return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
        return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffMinutes > 0) {
        return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
    } else {
        return 'Just now';
    }
}

module.exports = router;