/**
 * Neural Odyssey Vault Reveal Modal Component
 *
 * Dramatic reveal modal for newly unlocked vault items. Creates an engaging
 * "treasure chest opening" experience with animations, particle effects, and
 * detailed item presentation. Handles all vault item types and categories.
 *
 * Features:
 * - Cinematic reveal animations with particle effects
 * - Rarity-based visual themes and celebrations
 * - Category-specific presentations and icons
 * - Content preview with full reading capabilities
 * - Interactive rating and favorite systems
 * - Skill point rewards with animated counters
 * - Achievement integration and unlocks
 * - Social sharing and bookmarking
 * - Related content recommendations
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQueryClient } from 'react-query';
import {
    Sparkles,
    Star,
    Heart,
    BookOpen,
    Eye,
    EyeOff,
    Trophy,
    Award,
    Crown,
    Gem,
    Scroll,
    Brain,
    Lightbulb,
    Target,
    Zap,
    Flame,
    Shield,
    Lock,
    Unlock,
    ChevronRight,
    ChevronDown,
    ExternalLink,
    Download,
    Share2,
    Copy,
    Check,
    X,
    Volume2,
    VolumeX,
    Clock,
    Calendar,
    Tag,
    Bookmark,
    MessageSquare,
    ThumbsUp,
    ThumbsDown,
    MoreHorizontal
} from 'lucide-react';
import { api } from '../utils/api';
import toast from 'react-hot-toast';

const VaultRevealModal = ({ 
    vaultItem, 
    isOpen, 
    onClose, 
    trigger = 'auto',
    showCelebration = true,
    autoRead = true
}) => {
    const queryClient = useQueryClient();
    const modalRef = useRef(null);
    const [revealStage, setRevealStage] = useState('opening'); // 'opening' | 'revealed' | 'reading'
    const [showFullContent, setShowFullContent] = useState(false);
    const [userRating, setUserRating] = useState(vaultItem?.userRating || null);
    const [isFavorite, setIsFavorite] = useState(vaultItem?.isFavorite || false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [showParticles, setShowParticles] = useState(true);
    const [readingTime, setReadingTime] = useState(0);
    const [copied, setCopied] = useState(false);

    // Mark as read mutation
    const markAsReadMutation = useMutation(
        () => api.post(`/vault/items/${vaultItem.itemId}/read`),
        {
            onSuccess: () => {
                queryClient.invalidateQueries(['vaultItems']);
                if (autoRead) {
                    setRevealStage('reading');
                }
            }
        }
    );

    // Rating mutation
    const rateMutation = useMutation(
        ({ rating, notes }) => api.post(`/vault/items/${vaultItem.itemId}/rate`, { rating, notes }),
        {
            onSuccess: () => {
                toast.success('Rating saved!');
                queryClient.invalidateQueries(['vaultItems']);
            },
            onError: () => {
                toast.error('Failed to save rating');
            }
        }
    );

    // Favorite mutation
    const favoriteMutation = useMutation(
        () => api.post(`/vault/items/${vaultItem.itemId}/favorite`),
        {
            onSuccess: (data) => {
                setIsFavorite(data.data.isFavorite);
                toast.success(data.data.isFavorite ? 'Added to favorites!' : 'Removed from favorites');
                queryClient.invalidateQueries(['vaultItems']);
            },
            onError: () => {
                toast.error('Failed to update favorite status');
            }
        }
    );

    // Vault item configuration
    const categoryConfig = {
        secret_archives: {
            name: 'Secret Archives',
            icon: Brain,
            color: 'from-blue-500 to-blue-600',
            lightColor: 'from-blue-200 to-blue-300',
            particleColor: '#3B82F6',
            description: 'Hidden mathematical insights and historical discoveries'
        },
        controversy_files: {
            name: 'Controversy Files',
            icon: Flame,
            color: 'from-red-500 to-red-600',
            lightColor: 'from-red-200 to-red-300',
            particleColor: '#EF4444',
            description: 'Debates, failures, and controversial topics in AI/ML'
        },
        beautiful_mind: {
            name: 'Beautiful Mind',
            icon: Lightbulb,
            color: 'from-purple-500 to-purple-600',
            lightColor: 'from-purple-200 to-purple-300',
            particleColor: '#8B5CF6',
            description: 'Inspirational stories and breakthrough moments'
        }
    };

    const itemTypeConfig = {
        secret: {
            name: 'Secret',
            icon: Lock,
            description: 'Hidden knowledge and insider insights'
        },
        tool: {
            name: 'Tool',
            icon: Target,
            description: 'Practical utilities and advanced techniques'
        },
        artifact: {
            name: 'Artifact',
            icon: Scroll,
            description: 'Historical documents and breakthrough papers'
        },
        story: {
            name: 'Story',
            icon: BookOpen,
            description: 'Inspirational narratives and personal journeys'
        },
        wisdom: {
            name: 'Wisdom',
            icon: Crown,
            description: 'Deep philosophical insights and principles'
        }
    };

    const rarityConfig = {
        common: {
            name: 'Common',
            color: 'text-gray-400',
            bgColor: 'from-gray-500 to-gray-600',
            glowColor: 'shadow-gray-500/50',
            particles: 15,
            celebration: 'basic'
        },
        uncommon: {
            name: 'Uncommon',
            color: 'text-green-400',
            bgColor: 'from-green-500 to-green-600',
            glowColor: 'shadow-green-500/50',
            particles: 25,
            celebration: 'enhanced'
        },
        rare: {
            name: 'Rare',
            color: 'text-blue-400',
            bgColor: 'from-blue-500 to-blue-600',
            glowColor: 'shadow-blue-500/50',
            particles: 35,
            celebration: 'impressive'
        },
        epic: {
            name: 'Epic',
            color: 'text-purple-400',
            bgColor: 'from-purple-500 to-purple-600',
            glowColor: 'shadow-purple-500/50',
            particles: 50,
            celebration: 'spectacular'
        },
        legendary: {
            name: 'Legendary',
            color: 'text-yellow-400',
            bgColor: 'from-yellow-500 to-yellow-600',
            glowColor: 'shadow-yellow-500/50',
            particles: 75,
            celebration: 'legendary'
        }
    };

    const category = categoryConfig[vaultItem?.category] || categoryConfig.secret_archives;
    const itemType = itemTypeConfig[vaultItem?.itemType] || itemTypeConfig.secret;
    const rarity = rarityConfig[vaultItem?.rarity] || rarityConfig.common;

    // Calculate reading time
    useEffect(() => {
        if (vaultItem?.contentFull) {
            const words = vaultItem.contentFull.split(' ').length;
            const estimatedTime = Math.ceil(words / 200); // 200 WPM average
            setReadingTime(estimatedTime);
        }
    }, [vaultItem]);

    // Auto-progress reveal stages
    useEffect(() => {
        if (!isOpen || !vaultItem) return;

        const timer1 = setTimeout(() => {
            setRevealStage('revealed');
        }, 2000);

        const timer2 = setTimeout(() => {
            if (autoRead) {
                markAsReadMutation.mutate();
            }
        }, 3500);

        return () => {
            clearTimeout(timer1);
            clearTimeout(timer2);
        };
    }, [isOpen, vaultItem, autoRead]);

    // Handle rating
    const handleRating = (rating) => {
        setUserRating(rating);
        rateMutation.mutate({ rating, notes: '' });
    };

    // Handle favorite toggle
    const handleFavoriteToggle = () => {
        favoriteMutation.mutate();
    };

    // Handle copy to clipboard
    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(vaultItem.contentFull);
            setCopied(true);
            toast.success('Content copied to clipboard!');
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            toast.error('Failed to copy content');
        }
    };

    // Handle close
    const handleClose = () => {
        setRevealStage('opening');
        setShowFullContent(false);
        onClose();
    };

    // Particle component
    const Particle = ({ delay = 0, color = '#3B82F6' }) => (
        <motion.div
            initial={{ 
                opacity: 0, 
                scale: 0, 
                x: 0, 
                y: 0 
            }}
            animate={{ 
                opacity: [0, 1, 0], 
                scale: [0, 1, 0],
                x: Math.random() * 400 - 200,
                y: Math.random() * 400 - 200
            }}
            transition={{ 
                duration: 2, 
                delay,
                ease: "easeOut"
            }}
            className="absolute w-2 h-2 rounded-full pointer-events-none"
            style={{ backgroundColor: color }}
        />
    );

    // Render particles
    const renderParticles = () => {
        if (!showParticles || revealStage === 'opening') return null;

        return (
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
                {[...Array(rarity.particles)].map((_, index) => (
                    <Particle
                        key={index}
                        delay={index * 0.05}
                        color={category.particleColor}
                    />
                ))}
            </div>
        );
    };

    // Render unlock celebration
    const renderUnlockCelebration = () => {
        if (revealStage !== 'revealed') return null;

        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center mb-6"
            >
                <motion.div
                    animate={{ 
                        rotate: [0, 10, -10, 0],
                        scale: [1, 1.1, 1]
                    }}
                    transition={{ 
                        duration: 0.6,
                        repeat: 2
                    }}
                    className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-gradient-to-r ${rarity.bgColor} text-white`}
                >
                    <Trophy className="w-5 h-5" />
                    <span className="font-bold">Vault Item Unlocked!</span>
                </motion.div>
                
                {vaultItem.bonusPoints && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="mt-2 flex items-center justify-center space-x-1 text-yellow-400"
                    >
                        <Zap className="w-4 h-4" />
                        <span className="text-sm font-medium">+{vaultItem.bonusPoints} Skill Points</span>
                    </motion.div>
                )}
            </motion.div>
        );
    };

    // Render vault item header
    const renderItemHeader = () => {
        const CategoryIcon = category.icon;
        const ItemTypeIcon = itemType.icon;

        return (
            <div className="text-center mb-6">
                {/* Main Icon */}
                <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
                    className={`
                        inline-flex items-center justify-center w-20 h-20 rounded-full 
                        bg-gradient-to-br ${category.color} ${rarity.glowColor} shadow-2xl mb-4
                    `}
                >
                    <CategoryIcon className="w-10 h-10 text-white" />
                </motion.div>

                {/* Rarity Badge */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className={`inline-block px-3 py-1 rounded-full ${rarity.bgColor} text-white text-sm font-medium mb-2`}
                >
                    {rarity.name} {itemType.name}
                </motion.div>

                {/* Title */}
                <motion.h2
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    className="text-2xl font-bold text-white mb-2"
                >
                    {vaultItem.title}
                </motion.h2>

                {/* Category */}
                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.9 }}
                    className="text-gray-400 text-sm"
                >
                    {category.name} • {itemType.description}
                </motion.p>
            </div>
        );
    };

    // Render content preview
    const renderContentPreview = () => {
        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.1 }}
                className="mb-6"
            >
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold text-white">Preview</h3>
                        <div className="flex items-center space-x-2 text-sm text-gray-400">
                            <Clock className="w-4 h-4" />
                            <span>{readingTime} min read</span>
                        </div>
                    </div>
                    
                    <p className="text-gray-300 text-sm leading-relaxed">
                        {vaultItem.description}
                    </p>

                    {vaultItem.contentPreview && (
                        <div className="mt-4 pt-4 border-t border-gray-700">
                            <p className="text-gray-400 text-sm italic">
                                "{vaultItem.contentPreview}..."
                            </p>
                        </div>
                    )}
                </div>
            </motion.div>
        );
    };

    // Render full content
    const renderFullContent = () => {
        if (!showFullContent) return null;

        return (
            <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6"
            >
                <div className="bg-gray-900 rounded-lg p-6 border border-gray-600 max-h-96 overflow-y-auto">
                    <div className="prose prose-invert prose-sm max-w-none">
                        <div 
                            className="text-gray-300 leading-relaxed"
                            dangerouslySetInnerHTML={{ 
                                __html: vaultItem.contentFull?.replace(/\n/g, '<br>') 
                            }}
                        />
                    </div>
                </div>
            </motion.div>
        );
    };

    // Render action buttons
    const renderActionButtons = () => {
        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.3 }}
                className="space-y-4"
            >
                {/* Primary Actions */}
                <div className="flex items-center justify-center space-x-3">
                    <button
                        onClick={() => setShowFullContent(!showFullContent)}
                        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                    >
                        {showFullContent ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        <span>{showFullContent ? 'Hide Content' : 'Read Full Content'}</span>
                    </button>

                    <button
                        onClick={handleFavoriteToggle}
                        disabled={favoriteMutation.isLoading}
                        className={`
                            flex items-center space-x-2 px-4 py-3 rounded-lg font-medium transition-colors
                            ${isFavorite 
                                ? 'bg-red-600 hover:bg-red-700 text-white' 
                                : 'bg-gray-600 hover:bg-gray-700 text-white'
                            }
                        `}
                    >
                        <Heart className={`w-4 h-4 ${isFavorite ? 'fill-current' : ''}`} />
                        <span>{isFavorite ? 'Favorited' : 'Favorite'}</span>
                    </button>
                </div>

                {/* Rating */}
                <div className="text-center">
                    <p className="text-sm text-gray-400 mb-2">Rate this content:</p>
                    <div className="flex items-center justify-center space-x-1">
                        {[1, 2, 3, 4, 5].map((rating) => (
                            <button
                                key={rating}
                                onClick={() => handleRating(rating)}
                                disabled={rateMutation.isLoading}
                                className={`
                                    p-1 rounded transition-colors
                                    ${userRating >= rating ? 'text-yellow-400' : 'text-gray-500 hover:text-yellow-300'}
                                `}
                            >
                                <Star className={`w-5 h-5 ${userRating >= rating ? 'fill-current' : ''}`} />
                            </button>
                        ))}
                    </div>
                    {userRating && (
                        <p className="text-xs text-gray-500 mt-1">
                            You rated this {userRating}/5 stars
                        </p>
                    )}
                </div>

                {/* Secondary Actions */}
                <div className="flex items-center justify-center space-x-2">
                    {showFullContent && (
                        <button
                            onClick={handleCopy}
                            className="flex items-center space-x-1 text-gray-400 hover:text-white px-3 py-2 rounded text-sm transition-colors"
                        >
                            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                            <span>{copied ? 'Copied!' : 'Copy'}</span>
                        </button>
                    )}
                    
                    <button
                        onClick={() => setShowParticles(!showParticles)}
                        className="flex items-center space-x-1 text-gray-400 hover:text-white px-3 py-2 rounded text-sm transition-colors"
                    >
                        {showParticles ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                        <span>Effects</span>
                    </button>
                </div>
            </motion.div>
        );
    };

    // Render unlock trigger info
    const renderUnlockInfo = () => {
        if (!trigger || trigger === 'manual') return null;

        const triggerMessages = {
            'auto_unlock': 'Unlocked automatically based on your progress',
            'quest_completion': 'Unlocked by completing a quest',
            'phase_completion': 'Unlocked by completing a learning phase',
            'streak_milestone': 'Unlocked by maintaining your learning streak',
            'study_time_milestone': 'Unlocked by reaching a study time milestone'
        };

        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5 }}
                className="text-center mb-4"
            >
                <div className="inline-flex items-center space-x-2 bg-gray-800 px-4 py-2 rounded-full text-sm text-gray-400 border border-gray-700">
                    <Unlock className="w-4 h-4" />
                    <span>{triggerMessages[trigger] || 'Congratulations on unlocking this item!'}</span>
                </div>
            </motion.div>
        );
    };

    if (!isOpen || !vaultItem) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-4"
                onClick={handleClose}
            >
                <motion.div
                    ref={modalRef}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 200, damping: 20 }}
                    className="relative bg-gray-900 rounded-2xl p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-gray-700"
                    onClick={e => e.stopPropagation()}
                >
                    {/* Close Button */}
                    <button
                        onClick={handleClose}
                        className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors z-10"
                    >
                        <X className="w-6 h-6" />
                    </button>

                    {/* Background Glow */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${category.color} opacity-5 rounded-2xl`} />

                    {/* Particles */}
                    {renderParticles()}

                    {/* Content */}
                    <div className="relative">
                        {/* Opening Animation */}
                        {revealStage === 'opening' && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="text-center py-12"
                            >
                                <motion.div
                                    animate={{ 
                                        scale: [1, 1.2, 1],
                                        rotate: [0, 180, 360]
                                    }}
                                    transition={{ 
                                        duration: 2, 
                                        repeat: Infinity,
                                        ease: "easeInOut"
                                    }}
                                    className={`inline-block p-6 rounded-full bg-gradient-to-br ${category.color} mb-4`}
                                >
                                    <Lock className="w-12 h-12 text-white" />
                                </motion.div>
                                <h2 className="text-2xl font-bold text-white mb-2">Unlocking Vault...</h2>
                                <p className="text-gray-400">Revealing your treasure...</p>
                            </motion.div>
                        )}

                        {/* Revealed Content */}
                        {revealStage !== 'opening' && (
                            <>
                                {renderUnlockCelebration()}
                                {renderItemHeader()}
                                {renderUnlockInfo()}
                                {renderContentPreview()}
                                <AnimatePresence>
                                    {renderFullContent()}
                                </AnimatePresence>
                                {renderActionButtons()}
                            </>
                        )}
                    </div>

                    {/* Tags */}
                    {vaultItem.tags && vaultItem.tags.length > 0 && revealStage !== 'opening' && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 1.7 }}
                            className="mt-6 pt-6 border-t border-gray-700"
                        >
                            <div className="flex flex-wrap gap-2">
                                {vaultItem.tags.map((tag, index) => (
                                    <span
                                        key={index}
                                        className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full"
                                    >
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};

export default VaultRevealModal;