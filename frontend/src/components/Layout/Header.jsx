/**
 * Neural Odyssey Header Component
 *
 * Main navigation header for the Neural Learning Web platform. Provides
 * global navigation, user controls, search functionality, and contextual
 * information display. Responsive design with mobile optimization.
 *
 * Features:
 * - Responsive navigation with breadcrumbs
 * - Global search with intelligent suggestions
 * - User profile menu with quick actions
 * - Real-time notifications with dropdown
 * - Theme toggle and preferences
 * - Current learning context display
 * - Progress indicators and streak tracking
 * - Mobile-optimized hamburger menu
 * - Connection status and health monitoring
 *
 * Author: Neural Explorer
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
    Menu,
    Search,
    Bell,
    User,
    Settings,
    Sun,
    Moon,
    Monitor,
    ChevronDown,
    ChevronRight,
    Home,
    BookOpen,
    Target,
    Sparkles,
    BarChart3,
    Flame,
    Trophy,
    Clock,
    Zap,
    Brain,
    Star,
    LogOut,
    HelpCircle,
    Wifi,
    WifiOff,
    MapPin,
    Calendar,
    TrendingUp,
    CheckCircle,
    AlertCircle,
    X,
    Maximize2,
    Minimize2,
    RefreshCw,
    Command,
    ArrowRight,
    Plus,
    Filter,
    MoreHorizontal
} from 'lucide-react';
import { api } from '../../utils/api';
import toast from 'react-hot-toast';

const Header = ({
    onSidebarToggle,
    sidebarOpen,
    profile,
    theme = 'dark',
    onThemeChange,
    className = ''
}) => {
    const navigate = useNavigate();
    const location = useLocation();
    const queryClient = useQueryClient();
    
    // Refs for dropdowns
    const searchRef = useRef(null);
    const notificationRef = useRef(null);
    const profileRef = useRef(null);

    // State management
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [isNotificationOpen, setIsNotificationOpen] = useState(false);
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);

    // Fetch real-time data
    const { data: notificationsData } = useQuery(
        'headerNotifications',
        () => api.get('/notifications?limit=5'),
        {
            refetchInterval: 30000,
            retry: false
        }
    );

    const { data: progressData } = useQuery(
        'headerProgress',
        () => api.get('/learning/progress'),
        {
            refetchInterval: 60000,
            staleTime: 30000
        }
    );

    const { data: streakData } = useQuery(
        'headerStreak',
        () => api.get('/learning/streak'),
        {
            refetchInterval: 60000
        }
    );

    const { data: healthData } = useQuery(
        'headerHealth',
        () => api.get('/health'),
        {
            refetchInterval: 30000,
            retry: 1
        }
    );

    // Search mutation
    const searchMutation = useMutation(
        (query) => api.get(`/search?q=${encodeURIComponent(query)}&limit=8`),
        {
            onSuccess: (data) => {
                setSearchResults(data.data?.results || []);
                setIsSearching(false);
            },
            onError: () => {
                setSearchResults([]);
                setIsSearching(false);
            }
        }
    );

    // Mark notification as read mutation
    const markNotificationReadMutation = useMutation(
        (notificationId) => api.post(`/notifications/${notificationId}/read`),
        {
            onSuccess: () => {
                queryClient.invalidateQueries(['headerNotifications']);
            }
        }
    );

    // Breadcrumb configuration
    const getBreadcrumbs = () => {
        const path = location.pathname;
        const segments = path.split('/').filter(Boolean);
        
        const breadcrumbMap = {
            '': { label: 'Dashboard', icon: Home },
            'learning': { label: 'Learning Path', icon: BookOpen },
            'quests': { label: 'Quest Board', icon: Target },
            'vault': { label: 'Neural Vault', icon: Sparkles },
            'analytics': { label: 'Analytics', icon: BarChart3 },
            'settings': { label: 'Settings', icon: Settings }
        };

        const breadcrumbs = [{ label: 'Home', path: '/', icon: Home }];
        
        segments.forEach((segment, index) => {
            const path = '/' + segments.slice(0, index + 1).join('/');
            const config = breadcrumbMap[segment];
            
            if (config) {
                breadcrumbs.push({
                    label: config.label,
                    path,
                    icon: config.icon
                });
            }
        });

        return breadcrumbs;
    };

    // Handle search
    useEffect(() => {
        if (searchQuery.trim().length > 2) {
            setIsSearching(true);
            const timeoutId = setTimeout(() => {
                searchMutation.mutate(searchQuery);
            }, 300);
            return () => clearTimeout(timeoutId);
        } else {
            setSearchResults([]);
        }
    }, [searchQuery]);

    // Close dropdowns on outside click
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (searchRef.current && !searchRef.current.contains(event.target)) {
                setIsSearchOpen(false);
            }
            if (notificationRef.current && !notificationRef.current.contains(event.target)) {
                setIsNotificationOpen(false);
            }
            if (profileRef.current && !profileRef.current.contains(event.target)) {
                setIsProfileOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Handle keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (event) => {
            // Command/Ctrl + K for search
            if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
                event.preventDefault();
                setIsSearchOpen(true);
            }
            
            // Escape to close dropdowns
            if (event.key === 'Escape') {
                setIsSearchOpen(false);
                setIsNotificationOpen(false);
                setIsProfileOpen(false);
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);

    // Get theme icon
    const getThemeIcon = () => {
        switch (theme) {
            case 'light': return Sun;
            case 'dark': return Moon;
            default: return Monitor;
        }
    };

    // Render breadcrumbs
    const renderBreadcrumbs = () => {
        const breadcrumbs = getBreadcrumbs();
        
        return (
            <nav className="flex items-center space-x-2 text-sm">
                {breadcrumbs.map((crumb, index) => {
                    const IconComponent = crumb.icon;
                    const isLast = index === breadcrumbs.length - 1;
                    
                    return (
                        <div key={crumb.path} className="flex items-center space-x-2">
                            {index > 0 && (
                                <ChevronRight className="w-4 h-4 text-gray-500" />
                            )}
                            <button
                                onClick={() => navigate(crumb.path)}
                                className={`
                                    flex items-center space-x-1 px-2 py-1 rounded transition-colors
                                    ${isLast 
                                        ? 'text-white font-medium' 
                                        : 'text-gray-400 hover:text-white'
                                    }
                                `}
                            >
                                <IconComponent className="w-4 h-4" />
                                <span>{crumb.label}</span>
                            </button>
                        </div>
                    );
                })}
            </nav>
        );
    };

    // Render search dropdown
    const renderSearchDropdown = () => {
        if (!isSearchOpen) return null;

        return (
            <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full left-0 right-0 mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 max-h-96 overflow-y-auto"
            >
                {isSearching ? (
                    <div className="p-4 text-center">
                        <RefreshCw className="w-5 h-5 animate-spin text-blue-400 mx-auto mb-2" />
                        <p className="text-gray-400">Searching...</p>
                    </div>
                ) : searchResults.length > 0 ? (
                    <div className="py-2">
                        {searchResults.map((result, index) => (
                            <button
                                key={index}
                                onClick={() => {
                                    navigate(result.path || '/');
                                    setIsSearchOpen(false);
                                    setSearchQuery('');
                                }}
                                className="w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors flex items-center space-x-3"
                            >
                                <div className={`p-2 rounded-lg ${result.color || 'bg-gray-600'}`}>
                                    {result.icon && <result.icon className="w-4 h-4 text-white" />}
                                </div>
                                <div className="flex-1">
                                    <div className="font-medium text-white">{result.title}</div>
                                    <div className="text-sm text-gray-400">{result.description}</div>
                                </div>
                                <ArrowRight className="w-4 h-4 text-gray-500" />
                            </button>
                        ))}
                    </div>
                ) : searchQuery.length > 2 ? (
                    <div className="p-4 text-center text-gray-400">
                        <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>No results found for "{searchQuery}"</p>
                    </div>
                ) : (
                    <div className="p-4">
                        <div className="text-xs text-gray-500 mb-3">QUICK ACTIONS</div>
                        <div className="space-y-1">
                            {[
                                { label: 'Next Lesson', path: '/learning/next', icon: BookOpen },
                                { label: 'Random Quest', path: '/quests/random', icon: Target },
                                { label: 'Check Vault', path: '/vault/check', icon: Sparkles },
                                { label: 'View Analytics', path: '/analytics', icon: BarChart3 }
                            ].map((action, index) => (
                                <button
                                    key={index}
                                    onClick={() => {
                                        navigate(action.path);
                                        setIsSearchOpen(false);
                                    }}
                                    className="w-full px-3 py-2 text-left hover:bg-gray-700 rounded flex items-center space-x-2"
                                >
                                    <action.icon className="w-4 h-4 text-gray-400" />
                                    <span className="text-gray-300">{action.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </motion.div>
        );
    };

    // Render notifications dropdown
    const renderNotificationsDropdown = () => {
        if (!isNotificationOpen) return null;

        const notifications = notificationsData?.data?.notifications || [];
        const unreadCount = notifications.filter(n => !n.isRead).length;

        return (
            <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full right-0 mt-2 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 max-h-96 overflow-y-auto"
            >
                {/* Header */}
                <div className="px-4 py-3 border-b border-gray-700">
                    <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-white">Notifications</h3>
                        {unreadCount > 0 && (
                            <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded-full">
                                {unreadCount} new
                            </span>
                        )}
                    </div>
                </div>

                {/* Notifications List */}
                {notifications.length > 0 ? (
                    <div className="py-2">
                        {notifications.map((notification) => (
                            <div
                                key={notification.id}
                                className={`
                                    px-4 py-3 hover:bg-gray-700 transition-colors cursor-pointer
                                    ${!notification.isRead ? 'bg-blue-500 bg-opacity-10' : ''}
                                `}
                                onClick={() => {
                                    if (!notification.isRead) {
                                        markNotificationReadMutation.mutate(notification.id);
                                    }
                                    if (notification.path) {
                                        navigate(notification.path);
                                    }
                                    setIsNotificationOpen(false);
                                }}
                            >
                                <div className="flex items-start space-x-3">
                                    <div className={`p-2 rounded-lg ${notification.color || 'bg-blue-600'}`}>
                                        {notification.icon && <notification.icon className="w-4 h-4 text-white" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium text-white truncate">
                                            {notification.title}
                                        </div>
                                        <div className="text-sm text-gray-400 line-clamp-2">
                                            {notification.message}
                                        </div>
                                        <div className="text-xs text-gray-500 mt-1">
                                            {new Date(notification.createdAt).toLocaleDateString()}
                                        </div>
                                    </div>
                                    {!notification.isRead && (
                                        <div className="w-2 h-2 bg-blue-500 rounded-full mt-1" />
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="p-8 text-center text-gray-400">
                        <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p>No notifications yet</p>
                    </div>
                )}

                {/* Footer */}
                {notifications.length > 0 && (
                    <div className="px-4 py-3 border-t border-gray-700">
                        <button
                            onClick={() => {
                                navigate('/notifications');
                                setIsNotificationOpen(false);
                            }}
                            className="w-full text-center text-blue-400 hover:text-blue-300 text-sm"
                        >
                            View all notifications
                        </button>
                    </div>
                )}
            </motion.div>
        );
    };

    // Render profile dropdown
    const renderProfileDropdown = () => {
        if (!isProfileOpen) return null;

        const currentStreak = streakData?.data?.currentStreak || 0;
        const totalStudyTime = progressData?.data?.summary?.totalStudyTime || 0;
        const ThemeIcon = getThemeIcon();

        return (
            <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full right-0 mt-2 w-72 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50"
            >
                {/* Profile Info */}
                <div className="px-4 py-4 border-b border-gray-700">
                    <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                            <User className="w-6 h-6 text-white" />
                        </div>
                        <div className="flex-1">
                            <div className="font-semibold text-white">
                                {profile?.username || 'Neural Explorer'}
                            </div>
                            <div className="text-sm text-gray-400">
                                Phase {profile?.current_phase || 1} • Week {profile?.current_week || 1}
                            </div>
                        </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="mt-4 grid grid-cols-2 gap-3">
                        <div className="bg-gray-900 rounded-lg p-3 text-center">
                            <div className="flex items-center justify-center space-x-1 text-orange-400 mb-1">
                                <Flame className="w-4 h-4" />
                                <span className="font-bold">{currentStreak}</span>
                            </div>
                            <div className="text-xs text-gray-400">Day Streak</div>
                        </div>
                        <div className="bg-gray-900 rounded-lg p-3 text-center">
                            <div className="flex items-center justify-center space-x-1 text-green-400 mb-1">
                                <Clock className="w-4 h-4" />
                                <span className="font-bold">{Math.round(totalStudyTime / 60)}h</span>
                            </div>
                            <div className="text-xs text-gray-400">Study Time</div>
                        </div>
                    </div>
                </div>

                {/* Menu Items */}
                <div className="py-2">
                    {[
                        { 
                            label: 'View Profile', 
                            icon: User, 
                            path: '/profile',
                            description: 'Personal dashboard'
                        },
                        { 
                            label: 'Learning Analytics', 
                            icon: BarChart3, 
                            path: '/analytics',
                            description: 'Progress insights'
                        },
                        { 
                            label: 'Preferences', 
                            icon: Settings, 
                            path: '/settings',
                            description: 'App settings'
                        }
                    ].map((item, index) => (
                        <button
                            key={index}
                            onClick={() => {
                                navigate(item.path);
                                setIsProfileOpen(false);
                            }}
                            className="w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors flex items-center space-x-3"
                        >
                            <item.icon className="w-5 h-5 text-gray-400" />
                            <div className="flex-1">
                                <div className="text-white">{item.label}</div>
                                <div className="text-xs text-gray-400">{item.description}</div>
                            </div>
                        </button>
                    ))}
                </div>

                {/* Theme Toggle */}
                <div className="px-4 py-3 border-t border-gray-700">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-300">Theme</span>
                        <button
                            onClick={() => {
                                const themes = ['light', 'dark', 'system'];
                                const currentIndex = themes.indexOf(theme);
                                const nextTheme = themes[(currentIndex + 1) % themes.length];
                                onThemeChange?.(nextTheme);
                            }}
                            className="flex items-center space-x-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                        >
                            <ThemeIcon className="w-4 h-4" />
                            <span className="text-sm capitalize">{theme}</span>
                        </button>
                    </div>
                </div>

                {/* Sign Out */}
                <div className="px-4 py-3 border-t border-gray-700">
                    <button
                        onClick={() => {
                            // Handle sign out logic here
                            toast.success('Signed out successfully');
                            setIsProfileOpen(false);
                        }}
                        className="w-full flex items-center space-x-2 text-red-400 hover:text-red-300 transition-colors"
                    >
                        <LogOut className="w-4 h-4" />
                        <span>Sign Out</span>
                    </button>
                </div>
            </motion.div>
        );
    };

    // Render current context
    const renderCurrentContext = () => {
        const currentPhase = profile?.current_phase || 1;
        const currentWeek = profile?.current_week || 1;
        const currentStreak = streakData?.data?.currentStreak || 0;

        return (
            <div className="hidden lg:flex items-center space-x-4 text-sm">
                {/* Current Position */}
                <div className="flex items-center space-x-2 bg-gray-800 px-3 py-1 rounded-full">
                    <MapPin className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-300">P{currentPhase}W{currentWeek}</span>
                </div>

                {/* Streak */}
                <div className="flex items-center space-x-1 text-orange-400">
                    <Flame className="w-4 h-4" />
                    <span className="font-medium">{currentStreak}d</span>
                </div>

                {/* Connection Status */}
                <div className="flex items-center space-x-1">
                    {healthData?.data?.connected ? (
                        <div className="flex items-center space-x-1 text-green-400">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            <Wifi className="w-4 h-4" />
                        </div>
                    ) : (
                        <div className="flex items-center space-x-1 text-red-400">
                            <div className="w-2 h-2 bg-red-400 rounded-full" />
                            <WifiOff className="w-4 h-4" />
                        </div>
                    )}
                </div>
            </div>
        );
    };

    const notifications = notificationsData?.data?.notifications || [];
    const unreadNotifications = notifications.filter(n => !n.isRead).length;

    return (
        <header className={`
            sticky top-0 z-40 bg-gray-900 border-b border-gray-700 backdrop-blur-sm bg-opacity-95
            ${className}
        `}>
            <div className="px-4 lg:px-6">
                <div className="flex items-center justify-between h-16">
                    {/* Left Section */}
                    <div className="flex items-center space-x-4">
                        {/* Mobile Menu Toggle */}
                        <button
                            onClick={onSidebarToggle}
                            className="lg:hidden p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
                        >
                            <Menu className="w-5 h-5" />
                        </button>

                        {/* Breadcrumbs */}
                        <div className="hidden md:block">
                            {renderBreadcrumbs()}
                        </div>
                    </div>

                    {/* Center Section - Search */}
                    <div className="flex-1 max-w-md mx-4 relative" ref={searchRef}>
                        <div className="relative">
                            <input
                                type="text"
                                placeholder="Search lessons, quests, vault items..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onFocus={() => setIsSearchOpen(true)}
                                className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            />
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                            
                            {/* Search shortcut hint */}
                            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 hidden sm:flex items-center space-x-1 text-xs text-gray-500">
                                <Command className="w-3 h-3" />
                                <span>K</span>
                            </div>
                        </div>

                        {/* Search Dropdown */}
                        <AnimatePresence>
                            {renderSearchDropdown()}
                        </AnimatePresence>
                    </div>

                    {/* Right Section */}
                    <div className="flex items-center space-x-2">
                        {/* Current Context */}
                        {renderCurrentContext()}

                        {/* Notifications */}
                        <div className="relative" ref={notificationRef}>
                            <button
                                onClick={() => setIsNotificationOpen(!isNotificationOpen)}
                                className="relative p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
                            >
                                <Bell className="w-5 h-5" />
                                {unreadNotifications > 0 && (
                                    <div className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                                        <span className="text-xs text-white font-bold">
                                            {unreadNotifications > 9 ? '9+' : unreadNotifications}
                                        </span>
                                    </div>
                                )}
                            </button>

                            {/* Notifications Dropdown */}
                            <AnimatePresence>
                                {renderNotificationsDropdown()}
                            </AnimatePresence>
                        </div>

                        {/* Profile Menu */}
                        <div className="relative" ref={profileRef}>
                            <button
                                onClick={() => setIsProfileOpen(!isProfileOpen)}
                                className="flex items-center space-x-2 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
                            >
                                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                                    <User className="w-4 h-4 text-white" />
                                </div>
                                <ChevronDown className="w-4 h-4 hidden lg:block" />
                            </button>

                            {/* Profile Dropdown */}
                            <AnimatePresence>
                                {renderProfileDropdown()}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;