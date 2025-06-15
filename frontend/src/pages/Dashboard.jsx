/**
 * Neural Odyssey Dashboard Page
 *
 * Main dashboard interface providing comprehensive overview of learning progress,
 * achievements, analytics, and quick access to all major features. Serves as the
 * central hub for the Neural Learning Web platform.
 *
 * Features:
 * - Real-time progress tracking and analytics
 * - Interactive learning path visualization
 * - Quick action shortcuts and recommendations
 * - Recent activity timeline and notifications
 * - Study session management and streak tracking
 * - Vault rewards and achievement displays
 * - Responsive design with smooth animations
 * - Personalized content and insights
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';
import {
  Brain,
  TrendingUp,
  Target,
  Clock,
  Star,
  Award,
  ChevronRight,
  Play,
  BookOpen,
  Code,
  Zap,
  Trophy,
  Calendar,
  Flame,
  Eye,
  Lightbulb,
  ArrowRight,
  BarChart3,
  Map,
  Sparkles,
  Plus,
  RefreshCw,
  Settings,
  Bell,
  CheckCircle,
  Circle,
  Timer,
  Activity,
  TrendingDown,
  AlertTriangle,
  Coffee,
  Sunrise,
  Sun,
  Moon,
  Users,
  Globe,
  Bookmark,
  Filter,
  Search,
  MoreHorizontal
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import ProgressMap from '../components/Dashboard/ProgressMap';
import SkillTree from '../components/Dashboard/SkillTree';
import QuestCard from '../components/QuestCard';
import VaultRevealModal from '../components/VaultRevealModal';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const Dashboard = ({ profile: initialProfile }) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  // State management
  const [activeView, setActiveView] = useState('overview');
  const [selectedTimeRange, setSelectedTimeRange] = useState('week');
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [selectedVaultItem, setSelectedVaultItem] = useState(null);
  const [showVaultModal, setShowVaultModal] = useState(false);
  const [quickActionFilter, setQuickActionFilter] = useState('all');

  // Data fetching
  const { data: dashboardData, isLoading, refetch } = useQuery(
    ['dashboard', selectedTimeRange],
    () => api.get(`/analytics/dashboard?range=${selectedTimeRange}`),
    {
      refetchInterval: 30000,
      staleTime: 15000,
      retry: 3
    }
  );

  const { data: progressSummary } = useQuery(
    'progressSummary',
    () => api.get('/learning/progress'),
    { refetchInterval: 30000 }
  );

  const { data: todaySessions } = useQuery(
    'todaySessions',
    () => api.get('/learning/sessions/today'),
    { refetchInterval: 60000 }
  );

  const { data: upcomingItems } = useQuery(
    'upcomingItems',
    () => api.get('/learning/upcoming'),
    { refetchInterval: 60000 }
  );

  const { data: recentActivity } = useQuery(
    'recentActivity',
    () => api.get('/analytics/activity/recent'),
    { refetchInterval: 60000 }
  );

  const { data: vaultHighlights } = useQuery(
    'vaultHighlights',
    () => api.get('/vault/highlights'),
    { refetchInterval: 60000 }
  );

  const { data: recommendations } = useQuery(
    'recommendations',
    () => api.get('/learning/recommendations'),
    { refetchInterval: 300000 } // 5 minutes
  );

  // Mutations
  const startSessionMutation = useMutation(
    (sessionData) => api.post('/learning/sessions', sessionData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['todaySessions']);
        toast.success('Study session started!');
        setShowSessionModal(false);
      },
      onError: (error) => {
        toast.error('Failed to start session: ' + error.message);
      }
    }
  );

  // Get profile data
  const profile = initialProfile || progressSummary?.data?.profile;

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 24
      }
    }
  };

  // Helper functions
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  const getTimeOfDayGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return { text: 'Good morning', icon: Sunrise };
    if (hour < 17) return { text: 'Good afternoon', icon: Sun };
    return { text: 'Good evening', icon: Moon };
  };

  const getSessionTypeIcon = (type) => {
    switch (type) {
      case 'math': return Brain;
      case 'coding': return Code;
      case 'visual_projects': return Eye;
      case 'real_applications': return Target;
      default: return BookOpen;
    }
  };

  // Render hero section
  const renderHeroSection = () => {
    const greeting = getTimeOfDayGreeting();
    const GreetingIcon = greeting.icon;
    const currentStreak = profile?.current_streak_days || 0;
    const totalStudyTime = profile?.total_study_minutes || 0;

    return (
      <motion.div
        variants={itemVariants}
        className="bg-gradient-to-br from-blue-600 via-purple-600 to-purple-700 rounded-2xl p-8 text-white relative overflow-hidden"
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-neural-pattern opacity-20" />
        
        <div className="relative z-10">
          <div className="flex items-center justify-between mb-6">
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <GreetingIcon className="w-6 h-6 text-yellow-300" />
                <span className="text-lg font-medium opacity-90">{greeting.text}</span>
              </div>
              <h1 className="text-3xl font-bold">
                Welcome back, {profile?.username || 'Neural Explorer'}
              </h1>
              <p className="text-lg opacity-90 mt-1">
                Ready to continue your AI journey?
              </p>
            </div>
            
            <div className="text-right">
              <div className="flex items-center space-x-1 mb-2">
                <Flame className="w-5 h-5 text-orange-400" />
                <span className="text-2xl font-bold">{currentStreak}</span>
                <span className="text-sm opacity-90">day streak</span>
              </div>
              <div className="text-sm opacity-75">
                {formatTime(totalStudyTime)} total study time
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white bg-opacity-10 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <BookOpen className="w-5 h-5 text-green-300" />
                <span className="text-sm opacity-90">Current Phase</span>
              </div>
              <div className="text-2xl font-bold mt-1">
                Phase {profile?.current_phase || 1}
              </div>
            </div>
            
            <div className="bg-white bg-opacity-10 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-blue-300" />
                <span className="text-sm opacity-90">Week</span>
              </div>
              <div className="text-2xl font-bold mt-1">
                {profile?.current_week || 1}
              </div>
            </div>
            
            <div className="bg-white bg-opacity-10 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Trophy className="w-5 h-5 text-yellow-300" />
                <span className="text-sm opacity-90">Completed</span>
              </div>
              <div className="text-2xl font-bold mt-1">
                {progressSummary?.data?.summary?.completedLessons || 0}
              </div>
            </div>
            
            <div className="bg-white bg-opacity-10 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-5 h-5 text-purple-300" />
                <span className="text-sm opacity-90">Vault Items</span>
              </div>
              <div className="text-2xl font-bold mt-1">
                {vaultHighlights?.data?.totalUnlocked || 0}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Render today's focus section
  const renderTodaysFocus = () => {
    const todayData = todaySessions?.data || {};
    const recommendedType = todayData.recommended_next_type || 'math';
    const SessionIcon = getSessionTypeIcon(recommendedType);

    return (
      <motion.div
        variants={itemVariants}
        className="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            <span>Today's Focus</span>
          </h2>
          <button
            onClick={() => setShowSessionModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>Start Session</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Recommended Session */}
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-4 text-white">
            <div className="flex items-center space-x-2 mb-2">
              <SessionIcon className="w-5 h-5" />
              <span className="font-medium">Recommended</span>
            </div>
            <div className="text-lg font-bold capitalize">
              {recommendedType.replace('_', ' ')} Session
            </div>
            <div className="text-sm opacity-90 mt-1">
              {recommendedType === 'math' && 'Mathematical foundations'}
              {recommendedType === 'coding' && 'Programming practice'}
              {recommendedType === 'visual_projects' && 'Visual learning'}
              {recommendedType === 'real_applications' && 'Practical applications'}
            </div>
          </div>

          {/* Today's Progress */}
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-5 h-5 text-green-400" />
              <span className="font-medium text-white">Today's Progress</span>
            </div>
            <div className="text-lg font-bold text-white">
              {formatTime(todayData.total_minutes || 0)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {todayData.session_count || 0} sessions completed
            </div>
          </div>

          {/* Current Streak */}
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Flame className="w-5 h-5 text-orange-400" />
              <span className="font-medium text-white">Streak</span>
            </div>
            <div className="text-lg font-bold text-white">
              {profile?.current_streak_days || 0} days
            </div>
            <div className="text-sm text-gray-400 mt-1">
              Best: {profile?.longest_streak_days || 0} days
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Render learning overview
  const renderLearningOverview = () => {
    return (
      <motion.div
        variants={itemVariants}
        className="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Map className="w-5 h-5 text-green-400" />
            <span>Learning Overview</span>
          </h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveView(activeView === 'progress' ? 'skills' : 'progress')}
              className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg transition-colors flex items-center space-x-1"
            >
              {activeView === 'progress' ? <Brain className="w-4 h-4" /> : <Map className="w-4 h-4" />}
              <span>{activeView === 'progress' ? 'Skills' : 'Progress'}</span>
            </button>
          </div>
        </div>

        <div className="h-96">
          <AnimatePresence mode="wait">
            {activeView === 'progress' || activeView === 'overview' ? (
              <motion.div
                key="progress"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full"
              >
                <ProgressMap />
              </motion.div>
            ) : (
              <motion.div
                key="skills"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="h-full"
              >
                <SkillTree />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    );
  };

  // Render recent activity
  const renderRecentActivity = () => {
    const activities = recentActivity?.data?.activities || [];

    return (
      <motion.div
        variants={itemVariants}
        className="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <span>Recent Activity</span>
          </h2>
          <button
            onClick={() => navigate('/analytics')}
            className="text-purple-400 hover:text-purple-300 transition-colors flex items-center space-x-1 text-sm"
          >
            <span>View All</span>
            <ChevronRight className="w-3 h-3" />
          </button>
        </div>

        <div className="space-y-3 max-h-64 overflow-y-auto">
          {activities.length > 0 ? (
            activities.slice(0, 5).map((activity, index) => (
              <div
                key={index}
                className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
              >
                <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                  {activity.type === 'lesson_completed' && <CheckCircle className="w-4 h-4 text-white" />}
                  {activity.type === 'quest_completed' && <Trophy className="w-4 h-4 text-white" />}
                  {activity.type === 'vault_unlocked' && <Sparkles className="w-4 h-4 text-white" />}
                  {activity.type === 'streak_milestone' && <Flame className="w-4 h-4 text-white" />}
                </div>
                <div className="flex-1">
                  <div className="text-white font-medium">{activity.title}</div>
                  <div className="text-gray-400 text-sm">{activity.description}</div>
                </div>
                <div className="text-gray-500 text-xs">
                  {new Date(activity.timestamp).toLocaleDateString()}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-400">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No recent activity</p>
              <p className="text-sm">Start learning to see your progress here</p>
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  // Render quick actions
  const renderQuickActions = () => {
    const actions = [
      {
        id: 'next-lesson',
        title: 'Continue Learning',
        description: 'Resume your current lesson',
        icon: BookOpen,
        color: 'from-green-500 to-green-600',
        path: '/learning/next'
      },
      {
        id: 'practice-coding',
        title: 'Code Practice',
        description: 'Work on programming exercises',
        icon: Code,
        color: 'from-blue-500 to-blue-600',
        path: '/learning/coding'
      },
      {
        id: 'explore-vault',
        title: 'Explore Vault',
        description: 'Discover unlocked treasures',
        icon: Sparkles,
        color: 'from-purple-500 to-purple-600',
        path: '/vault'
      },
      {
        id: 'view-quests',
        title: 'Active Quests',
        description: 'Check your current challenges',
        icon: Target,
        color: 'from-orange-500 to-orange-600',
        path: '/quests'
      },
      {
        id: 'analytics',
        title: 'View Analytics',
        description: 'Track your progress',
        icon: BarChart3,
        color: 'from-cyan-500 to-cyan-600',
        path: '/analytics'
      },
      {
        id: 'settings',
        title: 'Settings',
        description: 'Customize your experience',
        icon: Settings,
        color: 'from-gray-500 to-gray-600',
        path: '/settings'
      }
    ];

    return (
      <motion.div
        variants={itemVariants}
        className="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          <span>Quick Actions</span>
        </h2>

        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          {actions.map((action) => {
            const IconComponent = action.icon;
            return (
              <button
                key={action.id}
                onClick={() => navigate(action.path)}
                className="group bg-gray-700 hover:bg-gray-600 rounded-lg p-4 transition-all duration-200 text-left"
              >
                <div className={`w-10 h-10 bg-gradient-to-br ${action.color} rounded-lg flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}>
                  <IconComponent className="w-5 h-5 text-white" />
                </div>
                <div className="text-white font-medium">{action.title}</div>
                <div className="text-gray-400 text-sm mt-1">{action.description}</div>
              </button>
            );
          })}
        </div>
      </motion.div>
    );
  };

  // Render recommendations
  const renderRecommendations = () => {
    const recommends = recommendations?.data || [];

    if (recommends.length === 0) return null;

    return (
      <motion.div
        variants={itemVariants}
        className="bg-gray-800 rounded-xl p-6 border border-gray-700"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
          <Lightbulb className="w-5 h-5 text-yellow-400" />
          <span>Recommended for You</span>
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {recommends.slice(0, 4).map((item, index) => (
            <div
              key={index}
              className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors cursor-pointer"
              onClick={() => navigate(item.path)}
            >
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center flex-shrink-0">
                  {item.type === 'lesson' && <BookOpen className="w-4 h-4 text-white" />}
                  {item.type === 'quest' && <Target className="w-4 h-4 text-white" />}
                  {item.type === 'review' && <RefreshCw className="w-4 h-4 text-white" />}
                </div>
                <div className="flex-1">
                  <div className="text-white font-medium">{item.title}</div>
                  <div className="text-gray-400 text-sm mt-1">{item.description}</div>
                  <div className="flex items-center space-x-2 mt-2">
                    <span className="text-xs bg-blue-500 text-white px-2 py-1 rounded-full">
                      {item.difficulty}
                    </span>
                    <span className="text-xs text-gray-400">
                      {item.estimatedTime}min
                    </span>
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-gray-400" />
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    );
  };

  // Loading state
  if (isLoading && !dashboardData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner 
          size="large" 
          text="Loading your Neural Odyssey..."
          variant="neural"
        />
      </div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-h-screen bg-gray-900 p-4 lg:p-6"
    >
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Hero Section */}
        {renderHeroSection()}

        {/* Today's Focus */}
        {renderTodaysFocus()}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Learning Overview */}
          <div className="lg:col-span-2">
            {renderLearningOverview()}
          </div>

          {/* Recent Activity */}
          {renderRecentActivity()}

          {/* Quick Actions */}
          {renderQuickActions()}
        </div>

        {/* Recommendations */}
        {renderRecommendations()}

        {/* Vault Reveal Modal */}
        <VaultRevealModal
          vaultItem={selectedVaultItem}
          isOpen={showVaultModal}
          onClose={() => {
            setShowVaultModal(false);
            setSelectedVaultItem(null);
          }}
        />
      </div>
    </motion.div>
  );
};

export default Dashboard;