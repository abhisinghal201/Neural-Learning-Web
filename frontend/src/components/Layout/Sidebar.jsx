/**
 * Neural Odyssey Sidebar Component
 *
 * Main navigation sidebar for the Neural Learning Web platform. Provides
 * navigation, user information, progress tracking, and quick access to
 * key features. Adapts between collapsed and expanded states with smooth
 * animations and responsive design.
 *
 * Features:
 * - Responsive navigation menu with React Router integration
 * - User profile display with avatar and progress indicators
 * - Real-time learning streak and statistics
 * - Quick access buttons for key features
 * - Health status monitoring and display
 * - Collapsible design with mobile optimization
 * - Animated state transitions
 * - Context-aware active states
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import {
    Home,
    BookOpen,
    Target,
    Sparkles,
    BarChart3,
    Settings,
    User,
    Flame,
    Trophy,
    Clock,
    TrendingUp,
    Brain,
    Code,
    Eye,
    Lightbulb,
    Zap,
    ChevronLeft,
    ChevronRight,
    Menu,
    X,
    Circle,
    CheckCircle,
    AlertCircle,
    Wifi,
    WifiOff,
    Crown,
    Star,
    Calendar,
    MapPin,
    Compass,
    Search,
    Bell,
    HelpCircle,
    LogOut,
    Minimize2,
    Maximize2
} from 'lucide-react';
import { api } from '../../utils/api';

const Sidebar = ({ 
    isOpen, 
    onClose, 
    profile, 
    healthStatus, 
    className = '',
    variant = 'default' // 'default' | 'minimal' | 'floating'
}) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [activeSection, setActiveSection] = useState('');

    // Fetch real-time data
    const { data: progressData } = useQuery(
        'sidebarProgress',
        () => api.get('/learning/progress'),
        {
            refetchInterval: 30000,
            staleTime: 15000
        }
    );

    const { data: streakData } = useQuery(
        'sidebarStreak',
        () => api.get('/learning/streak'),
        {
            refetchInterval: 60000
        }
    );

    const { data: vaultStats } = useQuery(
        'sidebarVault',
        () => api.get('/vault/statistics'),
        {
            refetchInterval: 60000
        }
    );

    const { data: notificationCount } = useQuery(
        'notificationCount',
        () => api.get('/notifications/count'),
        {
            refetchInterval: 30000,
            retry: false
        }
    );

    // Navigation configuration
    const navigationItems = [
        {
            id: 'dashboard',
            label: 'Dashboard',
            icon: Home,
            path: '/',
            description: 'Overview and progress',
            color: 'text-blue-400',
            badge: null
        },
        {
            id: 'learning',
            label: 'Learning Path',
            icon: BookOpen,
            path: '/learning',
            description: 'Structured curriculum',
            color: 'text-green-400',
            badge: progressData?.data?.summary?.inProgressLessons || null
        },
        {
            id: 'quests',
            label: 'Quest Board',
            icon: Target,
            path: '/quests',
            description: 'Challenges and projects',
            color: 'text-orange-400',
            badge: null
        },
        {
            id: 'vault',
            label: 'Neural Vault',
            icon: Sparkles,
            path: '/vault',
            description: 'Unlocked treasures',
            color: 'text-purple-400',
            badge: vaultStats?.data?.unlocked || null
        },
        {
            id: 'analytics',
            label: 'Analytics',
            icon: BarChart3,
            path: '/analytics',
            description: 'Progress insights',
            color: 'text-cyan-400',
            badge: null
        },
        {
            id: 'settings',
            label: 'Settings',
            icon: Settings,
            path: '/settings',
            description: 'Preferences and config',
            color: 'text-gray-400',
            badge: null
        }
    ];

    // Quick actions configuration
    const quickActions = [
        {
            id: 'next-lesson',
            label: 'Next Lesson',
            icon: ChevronRight,
            path: '/learning/next',
            color: 'bg-blue-600 hover:bg-blue-700'
        },
        {
            id: 'random-quest',
            label: 'Random Quest',
            icon: Target,
            path: '/quests/random',
            color: 'bg-green-600 hover:bg-green-700'
        },
        {
            id: 'check-vault',
            label: 'Check Vault',
            icon: Sparkles,
            action: () => api.post('/vault/check-unlocks'),
            color: 'bg-purple-600 hover:bg-purple-700'
        }
    ];

    // Determine active section based on current path
    useEffect(() => {
        const currentPath = location.pathname;
        const activeItem = navigationItems.find(item => 
            item.path === currentPath || 
            (item.path !== '/' && currentPath.startsWith(item.path))
        );
        setActiveSection(activeItem?.id || 'dashboard');
    }, [location.pathname]);

    // Handle navigation
    const handleNavigation = (path, action) => {
        if (action) {
            action();
        } else if (path) {
            navigate(path);
        }
        if (onClose) onClose();
    };

    // Render user profile section
    const renderUserProfile = () => {
        const currentStreak = streakData?.data?.currentStreak || 0;
        const totalStudyTime = progressData?.data?.summary?.totalStudyTime || 0;
        const completedLessons = progressData?.data?.summary?.completedLessons || 0;

        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`
                    p-4 border-b border-gray-700
                    ${isCollapsed ? 'px-2' : 'px-4'}
                `}
            >
                <div className="flex items-center space-x-3">
                    {/* Avatar */}
                    <div className="relative">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                            <User className="w-6 h-6 text-white" />
                        </div>
                        {/* Online status indicator */}
                        <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-gray-900" />
                    </div>

                    {/* User Info */}
                    <AnimatePresence>
                        {!isCollapsed && (
                            <motion.div
                                initial={{ opacity: 0, width: 0 }}
                                animate={{ opacity: 1, width: 'auto' }}
                                exit={{ opacity: 0, width: 0 }}
                                className="flex-1 min-w-0"
                            >
                                <h3 className="font-semibold text-white truncate">
                                    {profile?.username || 'Neural Explorer'}
                                </h3>
                                <div className="flex items-center space-x-2 text-xs text-gray-400">
                                    <div className="flex items-center space-x-1">
                                        <Flame className="w-3 h-3 text-orange-400" />
                                        <span>{currentStreak}d</span>
                                    </div>
                                    <span>•</span>
                                    <div className="flex items-center space-x-1">
                                        <Trophy className="w-3 h-3 text-yellow-400" />
                                        <span>{completedLessons}</span>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Quick Stats */}
                <AnimatePresence>
                    {!isCollapsed && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-3 grid grid-cols-2 gap-2"
                        >
                            <div className="bg-gray-800 rounded-lg p-2 text-center">
                                <div className="text-xs text-gray-400">Study Time</div>
                                <div className="text-sm font-semibold text-white">
                                    {Math.round(totalStudyTime / 60)}h
                                </div>
                            </div>
                            <div className="bg-gray-800 rounded-lg p-2 text-center">
                                <div className="text-xs text-gray-400">Current Phase</div>
                                <div className="text-sm font-semibold text-white">
                                    {profile?.current_phase || 1}
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        );
    };

    // Render navigation menu
    const renderNavigation = () => {
        return (
            <nav className="flex-1 py-4">
                <div className="space-y-1 px-2">
                    {navigationItems.map((item) => {
                        const isActive = activeSection === item.id;
                        const IconComponent = item.icon;

                        return (
                            <motion.button
                                key={item.id}
                                whileHover={{ x: 2 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => handleNavigation(item.path)}
                                className={`
                                    w-full flex items-center px-3 py-2.5 rounded-lg transition-all duration-200
                                    ${isActive 
                                        ? 'bg-blue-600 text-white shadow-lg' 
                                        : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                                    }
                                    ${isCollapsed ? 'justify-center' : 'justify-start'}
                                `}
                            >
                                <div className="relative">
                                    <IconComponent 
                                        className={`
                                            w-5 h-5 transition-colors
                                            ${isActive ? 'text-white' : item.color}
                                        `} 
                                    />
                                    {/* Badge */}
                                    {item.badge && (
                                        <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
                                            <span className="text-xs text-white font-bold">
                                                {item.badge > 9 ? '9+' : item.badge}
                                            </span>
                                        </div>
                                    )}
                                </div>

                                <AnimatePresence>
                                    {!isCollapsed && (
                                        <motion.div
                                            initial={{ opacity: 0, width: 0 }}
                                            animate={{ opacity: 1, width: 'auto' }}
                                            exit={{ opacity: 0, width: 0 }}
                                            className="ml-3 flex-1 text-left"
                                        >
                                            <div className="font-medium">{item.label}</div>
                                            <div className="text-xs opacity-75 truncate">
                                                {item.description}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                {/* Active indicator */}
                                {isActive && (
                                    <motion.div
                                        layoutId="activeIndicator"
                                        className="absolute left-0 w-1 h-8 bg-blue-400 rounded-r-full"
                                    />
                                )}
                            </motion.button>
                        );
                    })}
                </div>
            </nav>
        );
    };

    // Render quick actions
    const renderQuickActions = () => {
        return (
            <AnimatePresence>
                {!isCollapsed && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="px-4 py-3 border-t border-gray-700"
                    >
                        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                            Quick Actions
                        </h4>
                        <div className="space-y-2">
                            {quickActions.map((action) => {
                                const IconComponent = action.icon;
                                return (
                                    <button
                                        key={action.id}
                                        onClick={() => handleNavigation(action.path, action.action)}
                                        className={`
                                            w-full flex items-center space-x-2 px-3 py-2 rounded-lg
                                            text-white text-sm font-medium transition-colors
                                            ${action.color}
                                        `}
                                    >
                                        <IconComponent className="w-4 h-4" />
                                        <span>{action.label}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        );
    };

    // Render health status
    const renderHealthStatus = () => {
        const isHealthy = healthStatus?.status === 'healthy';
        const isConnected = healthStatus?.connected !== false;

        return (
            <AnimatePresence>
                {!isCollapsed && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="px-4 py-3 border-t border-gray-700"
                    >
                        <div className="flex items-center space-x-2">
                            {isConnected ? (
                                <div className="flex items-center space-x-2 text-green-400">
                                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                    <Wifi className="w-4 h-4" />
                                    <span className="text-xs">Connected</span>
                                </div>
                            ) : (
                                <div className="flex items-center space-x-2 text-red-400">
                                    <div className="w-2 h-2 bg-red-400 rounded-full" />
                                    <WifiOff className="w-4 h-4" />
                                    <span className="text-xs">Offline</span>
                                </div>
                            )}
                        </div>

                        {healthStatus?.dbSize && (
                            <div className="mt-1 text-xs text-gray-500">
                                DB: {healthStatus.dbSize}
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        );
    };

    // Render collapse toggle
    const renderCollapseToggle = () => {
        return (
            <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="absolute -right-3 top-8 w-6 h-6 bg-gray-800 border border-gray-600 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-700 transition-colors z-10"
            >
                {isCollapsed ? (
                    <ChevronRight className="w-3 h-3" />
                ) : (
                    <ChevronLeft className="w-3 h-3" />
                )}
            </button>
        );
    };

    // Main sidebar content
    const sidebarContent = (
        <motion.div
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className={`
                relative flex flex-col h-full bg-gray-900 border-r border-gray-700
                ${isCollapsed ? 'w-16' : 'w-64'}
                transition-all duration-300 ease-in-out
            `}
        >
            {/* Header */}
            <div className={`flex items-center ${isCollapsed ? 'justify-center px-2' : 'justify-between px-4'} py-4 border-b border-gray-700`}>
                <AnimatePresence>
                    {!isCollapsed && (
                        <motion.div
                            initial={{ opacity: 0, width: 0 }}
                            animate={{ opacity: 1, width: 'auto' }}
                            exit={{ opacity: 0, width: 0 }}
                            className="flex items-center space-x-2"
                        >
                            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                                <Brain className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h1 className="text-lg font-bold text-white">Neural Odyssey</h1>
                                <p className="text-xs text-gray-400">ML Learning Platform</p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Mobile close button */}
                {onClose && (
                    <button
                        onClick={onClose}
                        className="lg:hidden text-gray-400 hover:text-white transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                )}

                {/* Logo when collapsed */}
                {isCollapsed && (
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <Brain className="w-5 h-5 text-white" />
                    </div>
                )}
            </div>

            {/* User Profile */}
            {renderUserProfile()}

            {/* Navigation */}
            {renderNavigation()}

            {/* Quick Actions */}
            {renderQuickActions()}

            {/* Health Status */}
            {renderHealthStatus()}

            {/* Collapse Toggle (Desktop only) */}
            <div className="hidden lg:block">
                {renderCollapseToggle()}
            </div>
        </motion.div>
    );

    // Mobile overlay
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
        return (
            <AnimatePresence>
                {isOpen && (
                    <>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={onClose}
                            className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
                        />
                        {/* Sidebar */}
                        <div className={`fixed left-0 top-0 h-full z-50 lg:hidden ${className}`}>
                            {sidebarContent}
                        </div>
                    </>
                )}
            </AnimatePresence>
        );
    }

    // Desktop sidebar
    return (
        <div className={`sticky top-0 h-screen z-30 hidden lg:block ${className}`}>
            {sidebarContent}
        </div>
    );
};

export default Sidebar;