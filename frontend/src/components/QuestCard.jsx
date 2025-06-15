/**
 * Neural Odyssey Quest Card Component
 *
 * Interactive card component for displaying individual quests/projects in the
 * gamified learning system. Shows quest information, progress, requirements,
 * and provides action buttons for quest interaction.
 *
 * Features:
 * - Quest type visualization with icons and colors
 * - Difficulty indicators and time estimates
 * - Prerequisites and unlock status
 * - Progress tracking and completion states
 * - Skill point rewards and achievements
 * - Interactive actions (start, continue, complete)
 * - Responsive design with hover effects
 * - Integration with learning progress
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';
import {
    Code,
    Target,
    BookOpen,
    Lightbulb,
    Lock,
    Unlock,
    Play,
    CheckCircle,
    Clock,
    Star,
    Trophy,
    Zap,
    Award,
    ChevronRight,
    Eye,
    Brain,
    Flame,
    Sparkles,
    AlertCircle,
    Info,
    ExternalLink,
    Download,
    Upload,
    BarChart3,
    TrendingUp,
    Layers,
    Settings,
    Users,
    Calendar
} from 'lucide-react';
import { api } from '../utils/api';
import toast from 'react-hot-toast';

const QuestCard = ({ 
    quest, 
    className = '', 
    size = 'medium', // 'small' | 'medium' | 'large'
    showProgress = true,
    onClick,
    onStart,
    onContinue,
    onComplete
}) => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [isExpanded, setIsExpanded] = useState(false);
    const [showDetails, setShowDetails] = useState(false);

    // Quest completion mutation
    const startQuestMutation = useMutation(
        (questData) => api.post('/learning/quests', questData),
        {
            onSuccess: (data) => {
                toast.success('Quest started successfully!');
                queryClient.invalidateQueries(['quests']);
                queryClient.invalidateQueries(['learningProgress']);
                if (onStart) onStart(quest, data);
            },
            onError: (error) => {
                toast.error(error.response?.data?.message || 'Failed to start quest');
            }
        }
    );

    // Quest type configuration
    const questTypes = {
        coding_exercise: {
            name: 'Coding Exercise',
            icon: Code,
            color: 'from-green-500 to-green-600',
            lightColor: 'bg-green-100',
            description: 'Hands-on programming challenge',
            difficulty: 'Technical Implementation'
        },
        implementation_project: {
            name: 'Implementation Project',
            icon: Layers,
            color: 'from-blue-500 to-blue-600',
            lightColor: 'bg-blue-100',
            description: 'Complex multi-part project',
            difficulty: 'System Building'
        },
        theory_quiz: {
            name: 'Theory Quiz',
            icon: BookOpen,
            color: 'from-purple-500 to-purple-600',
            lightColor: 'bg-purple-100',
            description: 'Knowledge validation',
            difficulty: 'Conceptual Understanding'
        },
        practical_application: {
            name: 'Practical Application',
            icon: Target,
            color: 'from-orange-500 to-orange-600',
            lightColor: 'bg-orange-100',
            description: 'Real-world problem solving',
            difficulty: 'Applied Learning'
        }
    };

    // Difficulty configuration
    const difficultyConfig = {
        beginner: {
            label: 'Beginner',
            color: 'text-green-400',
            bgColor: 'bg-green-500 bg-opacity-20',
            stars: 1
        },
        intermediate: {
            label: 'Intermediate',
            color: 'text-yellow-400',
            bgColor: 'bg-yellow-500 bg-opacity-20',
            stars: 2
        },
        advanced: {
            label: 'Advanced',
            color: 'text-orange-400',
            bgColor: 'bg-orange-500 bg-opacity-20',
            stars: 3
        },
        expert: {
            label: 'Expert',
            color: 'text-red-400',
            bgColor: 'bg-red-500 bg-opacity-20',
            stars: 4
        }
    };

    // Status configuration
    const statusConfig = {
        locked: {
            label: 'Locked',
            color: 'text-gray-400',
            bgColor: 'bg-gray-500 bg-opacity-20',
            icon: Lock
        },
        available: {
            label: 'Available',
            color: 'text-blue-400',
            bgColor: 'bg-blue-500 bg-opacity-20',
            icon: Unlock
        },
        in_progress: {
            label: 'In Progress',
            color: 'text-yellow-400',
            bgColor: 'bg-yellow-500 bg-opacity-20',
            icon: Clock
        },
        attempted: {
            label: 'Attempted',
            color: 'text-orange-400',
            bgColor: 'bg-orange-500 bg-opacity-20',
            icon: Play
        },
        completed: {
            label: 'Completed',
            color: 'text-green-400',
            bgColor: 'bg-green-500 bg-opacity-20',
            icon: CheckCircle
        },
        mastered: {
            label: 'Mastered',
            color: 'text-purple-400',
            bgColor: 'bg-purple-500 bg-opacity-20',
            icon: Trophy
        }
    };

    // Get quest configuration
    const questConfig = questTypes[quest.type] || questTypes.coding_exercise;
    const difficultyInfo = difficultyConfig[quest.difficulty] || difficultyConfig.beginner;
    const statusInfo = statusConfig[quest.status] || statusConfig.available;

    // Size configuration
    const sizeConfig = {
        small: {
            cardClass: 'p-4',
            titleClass: 'text-base',
            descriptionClass: 'text-sm',
            iconSize: 'w-5 h-5',
            buttonClass: 'px-3 py-1 text-sm'
        },
        medium: {
            cardClass: 'p-6',
            titleClass: 'text-lg',
            descriptionClass: 'text-sm',
            iconSize: 'w-6 h-6',
            buttonClass: 'px-4 py-2'
        },
        large: {
            cardClass: 'p-8',
            titleClass: 'text-xl',
            descriptionClass: 'text-base',
            iconSize: 'w-8 h-8',
            buttonClass: 'px-6 py-3 text-lg'
        }
    };

    const sizeInfo = sizeConfig[size];

    // Calculate estimated reward points
    const calculateRewardPoints = () => {
        const basePoints = {
            coding_exercise: 15,
            implementation_project: 25,
            theory_quiz: 10,
            practical_application: 20
        };

        const difficultyMultiplier = {
            beginner: 1,
            intermediate: 1.5,
            advanced: 2,
            expert: 3
        };

        return Math.round((basePoints[quest.type] || 15) * (difficultyMultiplier[quest.difficulty] || 1));
    };

    // Handle quest action based on status
    const handleQuestAction = () => {
        if (onClick) {
            onClick(quest);
            return;
        }

        switch (quest.status) {
            case 'locked':
                // Show requirements
                setShowDetails(true);
                break;
            case 'available':
                // Start quest
                if (onStart) {
                    onStart(quest);
                } else {
                    navigate(`/quest/${quest.id}`);
                }
                break;
            case 'in_progress':
            case 'attempted':
                // Continue quest
                if (onContinue) {
                    onContinue(quest);
                } else {
                    navigate(`/quest/${quest.id}`);
                }
                break;
            case 'completed':
            case 'mastered':
                // View results
                navigate(`/quest/${quest.id}/results`);
                break;
            default:
                navigate(`/quest/${quest.id}`);
        }
    };

    // Get action button text and style
    const getActionButton = () => {
        switch (quest.status) {
            case 'locked':
                return {
                    text: 'View Requirements',
                    icon: Info,
                    variant: 'secondary'
                };
            case 'available':
                return {
                    text: 'Start Quest',
                    icon: Play,
                    variant: 'primary'
                };
            case 'in_progress':
            case 'attempted':
                return {
                    text: 'Continue',
                    icon: ChevronRight,
                    variant: 'primary'
                };
            case 'completed':
                return {
                    text: 'View Results',
                    icon: Eye,
                    variant: 'success'
                };
            case 'mastered':
                return {
                    text: 'Mastered!',
                    icon: Trophy,
                    variant: 'gold'
                };
            default:
                return {
                    text: 'View Quest',
                    icon: ChevronRight,
                    variant: 'secondary'
                };
        }
    };

    const actionButton = getActionButton();
    const QuestIcon = questConfig.icon;
    const StatusIcon = statusInfo.icon;
    const ActionIcon = actionButton.icon;

    // Render difficulty stars
    const renderDifficultyStars = () => {
        return (
            <div className="flex items-center space-x-1">
                {[...Array(4)].map((_, index) => (
                    <Star
                        key={index}
                        className={`w-3 h-3 ${
                            index < difficultyInfo.stars
                                ? `${difficultyInfo.color} fill-current`
                                : 'text-gray-600'
                        }`}
                    />
                ))}
            </div>
        );
    };

    // Render prerequisites
    const renderPrerequisites = () => {
        if (!quest.prerequisites || quest.prerequisites.length === 0) return null;

        return (
            <div className="mt-4 pt-4 border-t border-gray-700">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Prerequisites:</h4>
                <div className="space-y-1">
                    {quest.prerequisites.map((prereq, index) => (
                        <div key={index} className="flex items-center space-x-2 text-xs text-gray-400">
                            <AlertCircle className="w-3 h-3" />
                            <span>{prereq}</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    // Render learning objectives
    const renderLearningObjectives = () => {
        if (!quest.learningObjectives || quest.learningObjectives.length === 0) return null;

        return (
            <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-300 mb-2">Learning Objectives:</h4>
                <ul className="space-y-1">
                    {quest.learningObjectives.slice(0, isExpanded ? undefined : 3).map((objective, index) => (
                        <li key={index} className="flex items-start space-x-2 text-xs text-gray-400">
                            <Target className="w-3 h-3 mt-0.5 flex-shrink-0" />
                            <span>{objective}</span>
                        </li>
                    ))}
                </ul>
                {quest.learningObjectives.length > 3 && (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setIsExpanded(!isExpanded);
                        }}
                        className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center space-x-1"
                    >
                        <span>{isExpanded ? 'Show Less' : `+${quest.learningObjectives.length - 3} more`}</span>
                    </button>
                )}
            </div>
        );
    };

    // Render progress bar (if applicable)
    const renderProgress = () => {
        if (!showProgress || !quest.progress) return null;

        return (
            <div className="mt-4">
                <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-400">Progress</span>
                    <span className="text-xs text-gray-300">{Math.round(quest.progress * 100)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${quest.progress * 100}%` }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className={`h-1.5 rounded-full bg-gradient-to-r ${questConfig.color}`}
                    />
                </div>
            </div>
        );
    };

    // Render tags
    const renderTags = () => {
        if (!quest.tags || quest.tags.length === 0) return null;

        return (
            <div className="flex flex-wrap gap-1 mt-3">
                {quest.tags.slice(0, 3).map((tag, index) => (
                    <span
                        key={index}
                        className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded"
                    >
                        {tag}
                    </span>
                ))}
                {quest.tags.length > 3 && (
                    <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                        +{quest.tags.length - 3}
                    </span>
                )}
            </div>
        );
    };

    return (
        <>
            <motion.div
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ y: -4 }}
                transition={{ duration: 0.2 }}
                className={`
                    relative bg-gray-800 rounded-xl border border-gray-700 cursor-pointer
                    hover:border-gray-600 hover:shadow-xl transition-all duration-300
                    ${quest.status === 'locked' ? 'opacity-75' : ''}
                    ${className}
                `}
                onClick={handleQuestAction}
            >
                {/* Background Pattern */}
                <div className={`
                    absolute inset-0 bg-gradient-to-br ${questConfig.color} opacity-5 rounded-xl
                `} />

                {/* Content */}
                <div className={`relative ${sizeInfo.cardClass}`}>
                    {/* Header */}
                    <div className="flex items-start justify-between mb-4">
                        <div className="flex items-start space-x-3">
                            {/* Quest Icon */}
                            <div className={`
                                p-3 rounded-lg bg-gradient-to-br ${questConfig.color}
                                ${quest.status === 'locked' ? 'grayscale' : ''}
                            `}>
                                {quest.status === 'locked' ? (
                                    <Lock className={`${sizeInfo.iconSize} text-white`} />
                                ) : (
                                    <QuestIcon className={`${sizeInfo.iconSize} text-white`} />
                                )}
                            </div>

                            {/* Quest Info */}
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center space-x-2 mb-1">
                                    <h3 className={`font-bold text-white ${sizeInfo.titleClass} truncate`}>
                                        {quest.title}
                                    </h3>
                                    {quest.isNew && (
                                        <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                                            New
                                        </span>
                                    )}
                                </div>
                                <p className="text-xs text-gray-400 mb-2">{questConfig.name}</p>
                                <p className={`text-gray-300 ${sizeInfo.descriptionClass} line-clamp-2`}>
                                    {quest.description}
                                </p>
                            </div>
                        </div>

                        {/* Status Badge */}
                        <div className={`
                            flex items-center space-x-1 px-2 py-1 rounded-full ${statusInfo.bgColor}
                        `}>
                            <StatusIcon className={`w-3 h-3 ${statusInfo.color}`} />
                            <span className={`text-xs font-medium ${statusInfo.color}`}>
                                {statusInfo.label}
                            </span>
                        </div>
                    </div>

                    {/* Meta Information */}
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-4">
                            {/* Difficulty */}
                            <div className="flex items-center space-x-1">
                                {renderDifficultyStars()}
                                <span className={`text-xs ${difficultyInfo.color}`}>
                                    {difficultyInfo.label}
                                </span>
                            </div>

                            {/* Time Estimate */}
                            <div className="flex items-center space-x-1 text-gray-400">
                                <Clock className="w-3 h-3" />
                                <span className="text-xs">{quest.estimatedTime || 30}m</span>
                            </div>

                            {/* Phase/Week */}
                            {quest.phase && quest.week && (
                                <div className="flex items-center space-x-1 text-gray-400">
                                    <Calendar className="w-3 h-3" />
                                    <span className="text-xs">P{quest.phase}W{quest.week}</span>
                                </div>
                            )}
                        </div>

                        {/* Reward Points */}
                        <div className="flex items-center space-x-1 text-yellow-400">
                            <Zap className="w-3 h-3" />
                            <span className="text-xs font-medium">+{calculateRewardPoints()}</span>
                        </div>
                    </div>

                    {/* Progress */}
                    {renderProgress()}

                    {/* Learning Objectives */}
                    {renderLearningObjectives()}

                    {/* Prerequisites (if locked) */}
                    {quest.status === 'locked' && renderPrerequisites()}

                    {/* Tags */}
                    {renderTags()}

                    {/* Action Buttons */}
                    <div className="mt-6 pt-4 border-t border-gray-700">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                                {/* Completion Stats (if completed) */}
                                {(quest.status === 'completed' || quest.status === 'mastered') && quest.completedAt && (
                                    <div className="flex items-center space-x-2 text-xs text-gray-400">
                                        <CheckCircle className="w-3 h-3" />
                                        <span>
                                            Completed {new Date(quest.completedAt).toLocaleDateString()}
                                        </span>
                                    </div>
                                )}

                                {/* Attempts (if attempted) */}
                                {quest.attemptsCount > 0 && (
                                    <div className="flex items-center space-x-1 text-xs text-gray-400">
                                        <BarChart3 className="w-3 h-3" />
                                        <span>{quest.attemptsCount} attempts</span>
                                    </div>
                                )}
                            </div>

                            {/* Action Button */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleQuestAction();
                                }}
                                disabled={startQuestMutation.isLoading}
                                className={`
                                    flex items-center space-x-2 ${sizeInfo.buttonClass} rounded-lg
                                    font-medium transition-all duration-200 disabled:opacity-50
                                    ${actionButton.variant === 'primary' 
                                        ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                                        : actionButton.variant === 'success'
                                            ? 'bg-green-600 hover:bg-green-700 text-white'
                                            : actionButton.variant === 'gold'
                                                ? 'bg-yellow-500 hover:bg-yellow-600 text-black'
                                                : 'bg-gray-600 hover:bg-gray-700 text-white'
                                    }
                                `}
                            >
                                <ActionIcon className="w-4 h-4" />
                                <span>{actionButton.text}</span>
                            </button>
                        </div>
                    </div>

                    {/* Loading Overlay */}
                    {startQuestMutation.isLoading && (
                        <div className="absolute inset-0 bg-black bg-opacity-50 rounded-xl flex items-center justify-center">
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full"
                            />
                        </div>
                    )}
                </div>

                {/* Mastery Glow Effect */}
                {quest.status === 'mastered' && (
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-yellow-500 opacity-20 rounded-xl animate-pulse" />
                )}

                {/* Achievement Indicators */}
                {quest.achievements && quest.achievements.length > 0 && (
                    <div className="absolute -top-2 -right-2">
                        <div className="bg-yellow-500 rounded-full p-1">
                            <Trophy className="w-4 h-4 text-white" />
                        </div>
                    </div>
                )}
            </motion.div>

            {/* Quest Details Modal */}
            <AnimatePresence>
                {showDetails && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
                        onClick={() => setShowDetails(false)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-gray-900 rounded-xl p-6 max-w-md w-full border border-gray-700"
                            onClick={e => e.stopPropagation()}
                        >
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center space-x-3">
                                    <div className={`p-3 rounded-lg bg-gradient-to-br ${questConfig.color}`}>
                                        <QuestIcon className="w-6 h-6 text-white" />
                                    </div>
                                    <div>
                                        <h3 className="text-xl font-bold text-white">{quest.title}</h3>
                                        <p className="text-sm text-gray-400">{questConfig.name}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setShowDetails(false)}
                                    className="text-gray-400 hover:text-white"
                                >
                                    ×
                                </button>
                            </div>

                            {/* Content */}
                            <div className="space-y-4">
                                <p className="text-gray-300">{quest.description}</p>
                                
                                {renderLearningObjectives()}
                                {renderPrerequisites()}

                                <div className="pt-4 border-t border-gray-700">
                                    <button
                                        onClick={() => {
                                            setShowDetails(false);
                                            if (quest.status !== 'locked') {
                                                handleQuestAction();
                                            }
                                        }}
                                        className={`
                                            w-full py-2 px-4 rounded-lg font-medium transition-colors
                                            ${quest.status === 'locked'
                                                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                                                : 'bg-blue-600 hover:bg-blue-700 text-white'
                                            }
                                        `}
                                        disabled={quest.status === 'locked'}
                                    >
                                        {quest.status === 'locked' ? 'Complete Prerequisites First' : actionButton.text}
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default QuestCard;