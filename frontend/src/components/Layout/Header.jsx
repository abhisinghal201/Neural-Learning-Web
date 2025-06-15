import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  Menu,
  Search,
  Bell,
  Settings,
  User,
  Moon,
  Sun,
  Monitor,
  ChevronRight,
  Home,
  BookOpen,
  Target,
  Sparkles,
  BarChart3,
  Command,
  Wifi,
  WifiOff,
  Zap,
  Clock,
  TrendingUp,
  MapPin,
  Calendar,
  Brain,
  X,
  Star,
  Book,
  Code,
  Eye,
  LogOut,
  Bookmark,
  Download,
  HelpCircle
} from 'lucide-react';

// Utils
import { api } from '../../utils/api';

const Header = ({ 
  onSidebarToggle, 
  sidebarOpen, 
  profile, 
  theme, 
  onThemeChange,
  className = ''
}) => {
  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [isNotificationOpen, setIsNotificationOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  // Refs
  const searchRef = useRef(null);
  const notificationRef = useRef(null);
  const profileRef = useRef(null);

  // Hooks
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  // Fetch notifications
  const { data: notificationsData } = useQuery(
    'headerNotifications',
    () => api.get('/notifications'),
    {
      refetchInterval: 30000,
      retry: false,
      enabled: isOnline
    }
  );

  // Fetch system status
  const { data: systemStatus } = useQuery(
    'systemStatus',
    () => api.get('/health'),
    {
      refetchInterval: 60000,
      retry: false,
      enabled: isOnline
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
        toast.error('Search failed');
      }
    }
  );

  // Mark notification as read mutation
  const markNotificationReadMutation = useMutation(
    (notificationId) => api.patch(`/notifications/${notificationId}/read`),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('headerNotifications');
      }
    }
  );

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Generate breadcrumbs based on current route
  const getBreadcrumbs = () => {
    const pathSegments = location.pathname.split('/').filter(Boolean);
    
    const breadcrumbMap = {
      '': { label: 'Dashboard', icon: Home },
      'learning': { label: 'Learning Path', icon: BookOpen },
      'quests': { label: 'Quest Board', icon: Target },
      'vault': { label: 'Neural Vault', icon: Sparkles },
      'progress': { label: 'Progress Tracker', icon: BarChart3 },
      'analytics': { label: 'Analytics', icon: TrendingUp },
      'settings': { label: 'Settings', icon: Settings }
    };

    const breadcrumbs = [{ label: 'Neural Odyssey', path: '/', icon: Brain }];
    
    pathSegments.forEach((segment, index) => {
      const path = '/' + pathSegments.slice(0, index + 1).join('/');
      const config = breadcrumbMap[segment];
      
      if (config) {
        breadcrumbs.push({
          label: config.label,
          path,
          icon: config.icon
        });
      } else {
        // Dynamic routes like lesson/:id
        breadcrumbs.push({
          label: segment.charAt(0).toUpperCase() + segment.slice(1),
          path,
          icon: Eye
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
      setIsSearching(false);
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
        const searchInput = searchRef.current?.querySelector('input');
        searchInput?.focus();
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

  // Handle theme toggle
  const handleThemeToggle = () => {
    const themes = ['dark', 'light', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    onThemeChange(nextTheme);
    toast.success(`Theme: ${nextTheme}`, { duration: 1500 });
  };

  // Handle notification click
  const handleNotificationClick = (notification) => {
    if (!notification.isRead) {
      markNotificationReadMutation.mutate(notification.id);
    }
    
    // Navigate to related content if applicable
    if (notification.actionUrl) {
      navigate(notification.actionUrl);
    }
    
    setIsNotificationOpen(false);
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
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }
                `}
              >
                <IconComponent className="w-4 h-4" />
                <span className="hidden sm:inline">{crumb.label}</span>
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
            <div className="inline-flex items-center space-x-2 text-gray-400">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
              <span>Searching...</span>
            </div>
          </div>
        ) : searchResults.length > 0 ? (
          <div className="py-2">
            {searchResults.map((result, index) => (
              <button
                key={index}
                onClick={() => {
                  navigate(result.url);
                  setIsSearchOpen(false);
                  setSearchQuery('');
                }}
                className="w-full px-4 py-3 text-left hover:bg-gray-700 flex items-center space-x-3 transition-colors"
              >
                <div className="flex-shrink-0">
                  {result.type === 'lesson' && <Book className="w-4 h-4 text-blue-400" />}
                  {result.type === 'quest' && <Target className="w-4 h-4 text-green-400" />}
                  {result.type === 'vault' && <Sparkles className="w-4 h-4 text-purple-400" />}
                  {result.type === 'page' && <Eye className="w-4 h-4 text-gray-400" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{result.title}</p>
                  <p className="text-gray-400 text-sm truncate">{result.description}</p>
                </div>
                <div className="flex-shrink-0">
                  <ChevronRight className="w-4 h-4 text-gray-500" />
                </div>
              </button>
            ))}
          </div>
        ) : searchQuery.length > 2 ? (
          <div className="p-4 text-center text-gray-400">
            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No results found for "{searchQuery}"</p>
            <p className="text-xs mt-1">Try different keywords or check spelling</p>
          </div>
        ) : (
          <div className="p-4 text-center text-gray-400">
            <div className="space-y-2">
              <p className="font-medium">Quick Search</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button 
                  onClick={() => setSearchQuery('linear algebra')}
                  className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                >
                  Linear Algebra
                </button>
                <button 
                  onClick={() => setSearchQuery('neural networks')}
                  className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                >
                  Neural Networks
                </button>
                <button 
                  onClick={() => setSearchQuery('quests')}
                  className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                >
                  Quests
                </button>
                <button 
                  onClick={() => setSearchQuery('vault')}
                  className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                >
                  Vault Items
                </button>
              </div>
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

    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="absolute top-full right-0 mt-2 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 max-h-96 overflow-y-auto"
      >
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white">Notifications</h3>
            <button
              onClick={() => setIsNotificationOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {notifications.length === 0 ? (
          <div className="p-6 text-center text-gray-400">
            <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No new notifications</p>
            <p className="text-xs mt-1">You're all caught up!</p>
          </div>
        ) : (
          <div className="py-2">
            {notifications.map((notification) => (
              <button
                key={notification.id}
                onClick={() => handleNotificationClick(notification)}
                className={`
                  w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors
                  ${!notification.isRead ? 'bg-blue-900/20' : ''}
                `}
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {!notification.isRead && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium text-sm">{notification.title}</p>
                    <p className="text-gray-400 text-xs mt-1">{notification.message}</p>
                    <p className="text-gray-500 text-xs mt-1">{notification.timeAgo}</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}

        {notifications.length > 0 && (
          <div className="p-3 border-t border-gray-700">
            <button
              onClick={() => {
                navigate('/notifications');
                setIsNotificationOpen(false);
              }}
              className="w-full text-center text-blue-400 hover:text-blue-300 text-sm"
            >
              View All Notifications
            </button>
          </div>
        )}
      </motion.div>
    );
  };

  // Render profile dropdown
  const renderProfileDropdown = () => {
    if (!isProfileOpen) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="absolute top-full right-0 mt-2 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50"
      >
        {/* Profile Info */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-white font-medium">{profile?.username || 'Neural Explorer'}</p>
              <p className="text-gray-400 text-xs">
                Phase {profile?.current_phase || 1}, Week {profile?.current_week || 1}
              </p>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="p-4 border-b border-gray-700">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <p className="text-white font-bold text-lg">{profile?.current_streak_days || 0}</p>
              <p className="text-gray-400 text-xs">Day Streak</p>
            </div>
            <div>
              <p className="text-white font-bold text-lg">
                {Math.floor((profile?.total_study_minutes || 0) / 60)}h
              </p>
              <p className="text-gray-400 text-xs">Study Time</p>
            </div>
          </div>
        </div>

        {/* Menu Items */}
        <div className="py-2">
          <button
            onClick={() => {
              navigate('/progress');
              setIsProfileOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-gray-300 hover:text-white hover:bg-gray-700 flex items-center space-x-3"
          >
            <BarChart3 className="w-4 h-4" />
            <span>View Progress</span>
          </button>
          
          <button
            onClick={() => {
              navigate('/vault');
              setIsProfileOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-gray-300 hover:text-white hover:bg-gray-700 flex items-center space-x-3"
          >
            <Sparkles className="w-4 h-4" />
            <span>Neural Vault</span>
          </button>

          <button
            onClick={() => {
              navigate('/settings');
              setIsProfileOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-gray-300 hover:text-white hover:bg-gray-700 flex items-center space-x-3"
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>

          <hr className="my-2 border-gray-700" />

          <button
            onClick={() => {
              // Export progress
              window.open('/api/v1/learning/export', '_blank');
              setIsProfileOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-gray-300 hover:text-white hover:bg-gray-700 flex items-center space-x-3"
          >
            <Download className="w-4 h-4" />
            <span>Export Progress</span>
          </button>

          <button
            onClick={() => {
              navigate('/help');
              setIsProfileOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-gray-300 hover:text-white hover:bg-gray-700 flex items-center space-x-3"
          >
            <HelpCircle className="w-4 h-4" />
            <span>Help & Support</span>
          </button>
        </div>
      </motion.div>
    );
  };

  // Render current context indicator
  const renderCurrentContext = () => {
    const currentPhase = profile?.current_phase || 1;
    const currentWeek = profile?.current_week || 1;
    
    return (
      <div className="hidden lg:flex items-center space-x-4 text-sm">
        {/* Current Phase/Week */}
        <div className="flex items-center space-x-2 bg-gray-800 px-3 py-1 rounded-lg">
          <MapPin className="w-4 h-4 text-blue-400" />
          <span className="text-gray-300">
            Phase {currentPhase}, Week {currentWeek}
          </span>
        </div>

        {/* Study streak */}
        <div className="flex items-center space-x-2 bg-gray-800 px-3 py-1 rounded-lg">
          <Zap className="w-4 h-4 text-yellow-400" />
          <span className="text-gray-300">
            {profile?.current_streak_days || 0} day streak
          </span>
        </div>

        {/* Connection status */}
        <div className="flex items-center space-x-1">
          {isOnline ? (
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
  const ThemeIcon = getThemeIcon();

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
              aria-label="Toggle sidebar"
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
                className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-12 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              
              {/* Search shortcut hint */}
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 hidden sm:flex items-center space-x-1 text-xs text-gray-500">
                <Command className="w-3 h-3" />
                <span>K</span>
              </div>

              {/* Clear search */}
              {searchQuery && (
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                  }}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 sm:right-12 text-gray-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
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

            {/* Theme Toggle */}
            <button
              onClick={handleThemeToggle}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
              title={`Current theme: ${theme}`}
            >
              <ThemeIcon className="w-5 h-5" />
            </button>

            {/* Notifications */}
            <div className="relative" ref={notificationRef}>
              <button
                onClick={() => setIsNotificationOpen(!isNotificationOpen)}
                className="relative p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
                aria-label="Notifications"
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
                aria-label="Profile menu"
              >
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <span className="hidden sm:inline text-white font-medium">
                  {profile?.username || 'Explorer'}
                </span>
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