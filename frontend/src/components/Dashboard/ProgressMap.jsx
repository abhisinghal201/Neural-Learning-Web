/**
 * Neural Odyssey Progress Map Component
 *
 * Visual representation of the user's learning journey through the ML curriculum.
 * Displays a dynamic, interactive map showing progress through all 4 phases,
 * current position, completed lessons, upcoming challenges, and unlocked vault items.
 *
 * Features:
 * - 4-phase learning path visualization
 * - Interactive phase and week navigation
 * - Real-time progress tracking
 * - Completion status indicators
 * - Vault item unlock notifications
 * - Responsive design with smooth animations
 * - Streak and achievement displays
 * - Next lesson recommendations
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';
import {
    Map,
    MapPin,
    CheckCircle,
    Circle,
    Star,
    Lock,
    Unlock,
    Trophy,
    Flame,
    Brain,
    Code,
    Eye,
    Target,
    ChevronRight,
    ChevronDown,
    ChevronUp,
    Sparkles,
    Award,
    Calendar,
    Clock,
    TrendingUp,
    Zap,
    BookOpen,
    Lightbulb
} from 'lucide-react';
import { api } from '../../utils/api';

const ProgressMap = ({ profile, className = '' }) => {
    const navigate = useNavigate();
    const [selectedPhase, setSelectedPhase] = useState(null);
    const [hoveredWeek, setHoveredWeek] = useState(null);
    const [mapView, setMapView] = useState('overview'); // 'overview' | 'detailed'

    // Fetch learning progress data
    const { data: progressData, isLoading, error, refetch } = useQuery(
        'learningProgress',
        () => api.get('/learning/progress'),
        {
            refetchInterval: 30000,
            staleTime: 60000
        }
    );

    // Fetch vault statistics
    const { data: vaultData } = useQuery(
        'vaultProgress',
        () => api.get('/vault/statistics'),
        {
            refetchInterval: 60000
        }
    );

    // Fetch streak information
    const { data: streakData } = useQuery(
        'streakInfo',
        () => api.get('/learning/streak'),
        {
            refetchInterval: 30000
        }
    );

    // Process progress data
    const progressSummary = useMemo(() => {
        if (!progressData?.data) return null;

        const { summary, progressByPhase } = progressData.data;
        
        return {
            currentPhase: summary.currentPhase || 1,
            currentWeek: summary.currentWeek || 1,
            totalLessons: summary.totalLessons || 0,
            completedLessons: summary.completedLessons || 0,
            masteredLessons: summary.masteredLessons || 0,
            totalStudyTime: summary.totalStudyTime || 0,
            averageMasteryScore: summary.averageMasteryScore || 0,
            streakDays: summary.streakDays || 0,
            progressByPhase: progressByPhase || {}
        };
    }, [progressData]);

    // Phase configuration
    const phases = [
        {
            id: 1,
            title: 'Mathematical Foundations',
            description: 'Linear algebra, calculus, probability, and statistics',
            icon: Brain,
            color: 'from-blue-500 to-blue-600',
            lightColor: 'from-blue-100 to-blue-200',
            weeks: 12,
            concepts: ['Linear Algebra', 'Calculus', 'Probability', 'Statistics']
        },
        {
            id: 2,
            title: 'Core Machine Learning',
            description: 'Supervised learning, unsupervised learning, and evaluation',
            icon: Code,
            color: 'from-green-500 to-green-600',
            lightColor: 'from-green-100 to-green-200',
            weeks: 12,
            concepts: ['Supervised Learning', 'Unsupervised Learning', 'Model Evaluation', 'Feature Engineering']
        },
        {
            id: 3,
            title: 'Advanced Techniques',
            description: 'Deep learning, neural networks, and specialized algorithms',
            icon: Eye,
            color: 'from-purple-500 to-purple-600',
            lightColor: 'from-purple-100 to-purple-200',
            weeks: 12,
            concepts: ['Deep Learning', 'Neural Networks', 'Computer Vision', 'NLP']
        },
        {
            id: 4,
            title: 'Mastery & Applications',
            description: 'Real-world projects, optimization, and research',
            icon: Target,
            color: 'from-orange-500 to-orange-600',
            lightColor: 'from-orange-100 to-orange-200',
            weeks: 12,
            concepts: ['MLOps', 'Production Systems', 'Research Methods', 'Advanced Projects']
        }
    ];

    // Calculate phase statistics
    const getPhaseStats = (phaseId) => {
        if (!progressSummary?.progressByPhase[phaseId]) {
            return {
                totalLessons: 0,
                completedLessons: 0,
                completionRate: 0,
                weeks: {}
            };
        }

        const phaseData = progressSummary.progressByPhase[phaseId];
        let totalLessons = 0;
        let completedLessons = 0;

        Object.values(phaseData).forEach(week => {
            totalLessons += week.length;
            completedLessons += week.filter(lesson => 
                lesson.status === 'completed' || lesson.status === 'mastered'
            ).length;
        });

        return {
            totalLessons,
            completedLessons,
            completionRate: totalLessons > 0 ? completedLessons / totalLessons : 0,
            weeks: phaseData
        };
    };

    // Get current position indicator
    const getCurrentPosition = () => {
        if (!progressSummary) return { phase: 1, week: 1 };
        return {
            phase: progressSummary.currentPhase,
            week: progressSummary.currentWeek
        };
    };

    // Handle phase selection
    const handlePhaseClick = (phase) => {
        if (selectedPhase === phase.id) {
            setSelectedPhase(null);
            setMapView('overview');
        } else {
            setSelectedPhase(phase.id);
            setMapView('detailed');
        }
    };

    // Handle week navigation
    const handleWeekClick = (phaseId, weekNumber) => {
        navigate(`/learning/phase/${phaseId}/week/${weekNumber}`);
    };

    // Render phase card
    const renderPhaseCard = (phase) => {
        const stats = getPhaseStats(phase.id);
        const isSelected = selectedPhase === phase.id;
        const isCurrent = getCurrentPosition().phase === phase.id;
        const isCompleted = stats.completionRate >= 0.8;
        const isLocked = phase.id > getCurrentPosition().phase + 1;

        const IconComponent = phase.icon;

        return (
            <motion.div
                key={phase.id}
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: phase.id * 0.1 }}
                className={`relative cursor-pointer group ${isSelected ? 'z-10' : 'z-0'}`}
                onClick={() => !isLocked && handlePhaseClick(phase)}
            >
                <div className={`
                    relative overflow-hidden rounded-xl border-2 transition-all duration-300
                    ${isSelected ? 'border-white shadow-2xl scale-105' : 'border-gray-700 hover:border-gray-600'}
                    ${isCurrent ? 'ring-2 ring-yellow-400 ring-opacity-50' : ''}
                    ${isLocked ? 'opacity-50 cursor-not-allowed' : ''}
                `}>
                    {/* Background gradient */}
                    <div className={`
                        absolute inset-0 bg-gradient-to-br opacity-90
                        ${isLocked ? 'from-gray-600 to-gray-700' : phase.color}
                    `} />

                    {/* Content */}
                    <div className="relative p-6 text-white">
                        {/* Header */}
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center space-x-3">
                                <div className={`
                                    p-3 rounded-lg bg-white bg-opacity-20 backdrop-blur-sm
                                    ${isCurrent ? 'ring-2 ring-yellow-300' : ''}
                                `}>
                                    {isLocked ? (
                                        <Lock className="w-6 h-6" />
                                    ) : isCompleted ? (
                                        <CheckCircle className="w-6 h-6 text-green-300" />
                                    ) : (
                                        <IconComponent className="w-6 h-6" />
                                    )}
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold">Phase {phase.id}</h3>
                                    <p className="text-sm opacity-90">{phase.title}</p>
                                </div>
                            </div>

                            {/* Current indicator */}
                            {isCurrent && (
                                <div className="flex items-center space-x-1 bg-yellow-500 bg-opacity-20 px-2 py-1 rounded-full">
                                    <MapPin className="w-4 h-4 text-yellow-300" />
                                    <span className="text-xs font-medium">Current</span>
                                </div>
                            )}
                        </div>

                        {/* Description */}
                        <p className="text-sm mb-4 opacity-90">{phase.description}</p>

                        {/* Progress bar */}
                        <div className="mb-4">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-xs opacity-75">Progress</span>
                                <span className="text-xs font-medium">
                                    {Math.round(stats.completionRate * 100)}%
                                </span>
                            </div>
                            <div className="w-full bg-white bg-opacity-20 rounded-full h-2">
                                <div
                                    className="bg-white h-2 rounded-full transition-all duration-500"
                                    style={{ width: `${stats.completionRate * 100}%` }}
                                />
                            </div>
                        </div>

                        {/* Stats */}
                        <div className="flex justify-between text-xs opacity-90">
                            <span>{stats.completedLessons}/{stats.totalLessons} lessons</span>
                            <span>{phase.weeks} weeks</span>
                        </div>

                        {/* Concepts preview */}
                        <div className="mt-3 flex flex-wrap gap-1">
                            {phase.concepts.slice(0, 2).map((concept, index) => (
                                <span
                                    key={index}
                                    className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded"
                                >
                                    {concept}
                                </span>
                            ))}
                            {phase.concepts.length > 2 && (
                                <span className="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">
                                    +{phase.concepts.length - 2} more
                                </span>
                            )}
                        </div>

                        {/* Expand indicator */}
                        {!isLocked && (
                            <div className="absolute bottom-2 right-2">
                                {isSelected ? (
                                    <ChevronUp className="w-4 h-4 opacity-60" />
                                ) : (
                                    <ChevronDown className="w-4 h-4 opacity-60" />
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Week details (expanded view) */}
                <AnimatePresence>
                    {isSelected && !isLocked && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                            className="mt-4 bg-gray-800 rounded-lg p-4 border border-gray-700"
                        >
                            <h4 className="text-sm font-semibold text-white mb-3">Week Progress</h4>
                            <div className="grid grid-cols-4 gap-2">
                                {Array.from({ length: phase.weeks }, (_, index) => {
                                    const weekNumber = index + 1;
                                    const weekData = stats.weeks[weekNumber] || [];
                                    const weekCompletion = weekData.length > 0 
                                        ? weekData.filter(lesson => 
                                            lesson.status === 'completed' || lesson.status === 'mastered'
                                          ).length / weekData.length 
                                        : 0;
                                    const isCurrentWeek = isCurrent && getCurrentPosition().week === weekNumber;

                                    return (
                                        <motion.div
                                            key={weekNumber}
                                            whileHover={{ scale: 1.05 }}
                                            className={`
                                                relative p-2 rounded cursor-pointer transition-all duration-200
                                                ${isCurrentWeek 
                                                    ? 'bg-yellow-500 bg-opacity-20 border border-yellow-400' 
                                                    : weekCompletion >= 1 
                                                        ? 'bg-green-500 bg-opacity-20 border border-green-400'
                                                        : weekCompletion > 0
                                                            ? 'bg-blue-500 bg-opacity-20 border border-blue-400'
                                                            : 'bg-gray-700 border border-gray-600'
                                                }
                                                hover:bg-opacity-30
                                            `}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleWeekClick(phase.id, weekNumber);
                                            }}
                                            onMouseEnter={() => setHoveredWeek({ phase: phase.id, week: weekNumber })}
                                            onMouseLeave={() => setHoveredWeek(null)}
                                        >
                                            <div className="text-center">
                                                <div className="text-xs font-medium text-white">
                                                    Week {weekNumber}
                                                </div>
                                                <div className="text-xs text-gray-300 mt-1">
                                                    {Math.round(weekCompletion * 100)}%
                                                </div>
                                            </div>

                                            {/* Current week indicator */}
                                            {isCurrentWeek && (
                                                <div className="absolute -top-1 -right-1">
                                                    <MapPin className="w-3 h-3 text-yellow-400" />
                                                </div>
                                            )}

                                            {/* Completion indicator */}
                                            {weekCompletion >= 1 && (
                                                <div className="absolute -top-1 -right-1">
                                                    <CheckCircle className="w-3 h-3 text-green-400" />
                                                </div>
                                            )}
                                        </motion.div>
                                    );
                                })}
                            </div>

                            {/* Week hover tooltip */}
                            {hoveredWeek && hoveredWeek.phase === phase.id && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="mt-3 p-3 bg-gray-900 rounded-lg border border-gray-600"
                                >
                                    <p className="text-sm text-white">
                                        Week {hoveredWeek.week} - Click to explore lessons and quests
                                    </p>
                                </motion.div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        );
    };

    // Render summary statistics
    const renderSummaryStats = () => {
        if (!progressSummary) return null;

        const completionRate = progressSummary.totalLessons > 0 
            ? progressSummary.completedLessons / progressSummary.totalLessons 
            : 0;

        const masteryRate = progressSummary.completedLessons > 0 
            ? progressSummary.masteredLessons / progressSummary.completedLessons 
            : 0;

        return (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                {/* Overall Progress */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                >
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-blue-500 bg-opacity-20 rounded-lg">
                            <TrendingUp className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">
                                {Math.round(completionRate * 100)}%
                            </p>
                            <p className="text-sm text-gray-400">Overall Progress</p>
                        </div>
                    </div>
                </motion.div>

                {/* Learning Streak */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 }}
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                >
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-orange-500 bg-opacity-20 rounded-lg">
                            <Flame className="w-5 h-5 text-orange-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">
                                {streakData?.data?.currentStreak || 0}
                            </p>
                            <p className="text-sm text-gray-400">Day Streak</p>
                        </div>
                    </div>
                </motion.div>

                {/* Study Time */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                >
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-green-500 bg-opacity-20 rounded-lg">
                            <Clock className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">
                                {Math.round(progressSummary.totalStudyTime / 60)}h
                            </p>
                            <p className="text-sm text-gray-400">Study Time</p>
                        </div>
                    </div>
                </motion.div>

                {/* Vault Items */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                >
                    <div className="flex items-center space-x-3">
                        <div className="p-2 bg-purple-500 bg-opacity-20 rounded-lg">
                            <Sparkles className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">
                                {vaultData?.data?.unlocked || 0}
                            </p>
                            <p className="text-sm text-gray-400">Vault Items</p>
                        </div>
                    </div>
                </motion.div>
            </div>
        );
    };

    // Render current position indicator
    const renderCurrentPosition = () => {
        if (!progressSummary) return null;

        const currentPos = getCurrentPosition();
        const currentPhase = phases.find(p => p.id === currentPos.phase);

        if (!currentPhase) return null;

        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 p-6 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-xl text-white"
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <div className="p-3 bg-white bg-opacity-20 rounded-full">
                            <MapPin className="w-6 h-6" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold">Current Position</h3>
                            <p className="opacity-90">
                                Phase {currentPos.phase}: {currentPhase.title} - Week {currentPos.week}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={() => navigate(`/learning/phase/${currentPos.phase}/week/${currentPos.week}`)}
                        className="flex items-center space-x-2 bg-white bg-opacity-20 hover:bg-opacity-30 px-4 py-2 rounded-lg transition-all duration-200"
                    >
                        <span>Continue Learning</span>
                        <ChevronRight className="w-4 h-4" />
                    </button>
                </div>
            </motion.div>
        );
    };

    // Loading state
    if (isLoading) {
        return (
            <div className={`space-y-6 ${className}`}>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="bg-gray-800 rounded-lg p-4 animate-pulse">
                            <div className="h-16 bg-gray-700 rounded"></div>
                        </div>
                    ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="bg-gray-800 rounded-xl p-6 animate-pulse">
                            <div className="h-32 bg-gray-700 rounded"></div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className={`text-center py-12 ${className}`}>
                <div className="text-red-400 mb-4">
                    <AlertTriangle className="w-12 h-12 mx-auto" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Failed to load progress</h3>
                <p className="text-gray-400 mb-4">Unable to fetch your learning progress.</p>
                <button
                    onClick={() => refetch()}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return (
        <div className={`space-y-6 ${className}`}>
            {/* Summary Statistics */}
            {renderSummaryStats()}

            {/* Current Position */}
            {renderCurrentPosition()}

            {/* Progress Map Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white mb-2">Learning Progress Map</h2>
                    <p className="text-gray-400">
                        Track your journey through the Neural Odyssey ML curriculum
                    </p>
                </div>
                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => setMapView(mapView === 'overview' ? 'detailed' : 'overview')}
                        className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg transition-colors"
                    >
                        <Map className="w-4 h-4" />
                        <span>{mapView === 'overview' ? 'Detailed View' : 'Overview'}</span>
                    </button>
                </div>
            </div>

            {/* Phase Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {phases.map(phase => renderPhaseCard(phase))}
            </div>

            {/* Quick Actions */}
            <div className="mt-8 p-6 bg-gray-800 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <button
                        onClick={() => navigate('/learning/next-lesson')}
                        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg transition-colors"
                    >
                        <BookOpen className="w-4 h-4" />
                        <span>Next Lesson</span>
                    </button>
                    <button
                        onClick={() => navigate('/quests')}
                        className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white p-3 rounded-lg transition-colors"
                    >
                        <Target className="w-4 h-4" />
                        <span>View Quests</span>
                    </button>
                    <button
                        onClick={() => navigate('/vault')}
                        className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white p-3 rounded-lg transition-colors"
                    >
                        <Sparkles className="w-4 h-4" />
                        <span>Vault</span>
                    </button>
                    <button
                        onClick={() => navigate('/analytics')}
                        className="flex items-center space-x-2 bg-orange-600 hover:bg-orange-700 text-white p-3 rounded-lg transition-colors"
                    >
                        <TrendingUp className="w-4 h-4" />
                        <span>Analytics</span>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ProgressMap;