/**
 * Enhanced Neural Odyssey Dashboard Page
 *
 * Now includes spaced repetition system integration using the confirmed backend capabilities:
 * - SM-2 algorithm implementation
 * - Concept review scheduling
 * - Difficulty factor adjustments
 * - Review performance tracking
 * 
 * Features added:
 * - Review widget showing items due today
 * - Quick review interface with quality scoring
 * - Review streak tracking
 * - Concept mastery progression
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
  MoreHorizontal,
  RotateCcw,
  Check,
  X,
  ChevronUp,
  ChevronDown,
  Archive,
  Repeat,
  Layers,
  MessageSquare
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

  // Spaced Repetition State
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [currentReviewItem, setCurrentReviewItem] = useState(null);
  const [reviewQueue, setReviewQueue] = useState([]);
  const [reviewIndex, setReviewIndex] = useState(0);
  const [selectedQuality, setSelectedQuality] = useState(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [reviewStreak, setReviewStreak] = useState(0);

  // Data fetching
  const { data: dashboardData, isLoading, refetch } = useQuery(
    ['dashboard', selectedTimeRange],
    () => api.get(`/learning/analytics?timeframe=${selectedTimeRange === 'week' ? 7 : selectedTimeRange === 'month' ? 30 : 90}`),
    {
      refetchInterval: 30000,
      staleTime: 15000,
      retry: 3
    }
  );

  const { data: progressData, isLoading: progressLoading } = useQuery(
    'dashboardProgress',
    () => api.get('/learning/progress'),
    {
      refetchInterval: 60000,
      staleTime: 30000
    }
  );

  // Spaced Repetition data
  const { data: reviewData, isLoading: reviewLoading } = useQuery(
    'spacedRepetitionReview',
    () => api.get('/learning/spaced-repetition'),
    {
      refetchInterval: 300000, // 5 minutes
      staleTime: 60000,
      onSuccess: (data) => {
        setReviewQueue(data.data?.review_items || []);
      }
    }
  );

  const { data: todaySessionsData } = useQuery(
    'todaySessions',
    () => api.get('/learning/sessions/today'),
    {
      refetchInterval: 60000
    }
  );

  const { data: vaultPreviewData } = useQuery(
    'vaultPreview',
    () => api.get('/vault/recent?limit=3'),
    {
      refetchInterval: 300000
    }
  );

  // Review submission mutation
  const submitReviewMutation = useMutation(
    ({ conceptId, qualityScore }) => 
      api.post(`/learning/spaced-repetition/${conceptId}/review`, { quality_score: qualityScore }),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('spacedRepetitionReview');
        
        // Update review streak
        if (selectedQuality >= 3) {
          setReviewStreak(prev => prev + 1);
        } else {
          setReviewStreak(0);
        }

        // Show feedback
        const feedback = selectedQuality >= 4 ? 'Excellent!' : 
                        selectedQuality >= 3 ? 'Good work!' : 
                        'Keep practicing!';
        toast.success(feedback);

        // Move to next item or finish
        if (reviewIndex < reviewQueue.length - 1) {
          setReviewIndex(prev => prev + 1);
          setCurrentReviewItem(reviewQueue[reviewIndex + 1]);
          setShowAnswer(false);
          setSelectedQuality(null);
        } else {
          // Finished all reviews
          const completed = reviewIndex + 1;
          toast.success(`Review session complete! ${completed} concept${completed > 1 ? 's' : ''} reviewed.`);
          setShowReviewModal(false);
          resetReviewState();
        }
      },
      onError: () => {
        toast.error('Failed to submit review');
      }
    }
  );

  // Reset review state
  const resetReviewState = () => {
    setCurrentReviewItem(null);
    setReviewIndex(0);
    setSelectedQuality(null);
    setShowAnswer(false);
  };

  // Start review session
  const startReviewSession = () => {
    if (reviewQueue.length === 0) {
      toast.info('No items to review today. Great job staying on top of your studies!');
      return;
    }
    
    setCurrentReviewItem(reviewQueue[0]);
    setReviewIndex(0);
    setShowReviewModal(true);
    setShowAnswer(false);
    setSelectedQuality(null);
  };

  // Handle review submission
  const handleReviewSubmit = () => {
    if (selectedQuality === null) {
      toast.error('Please select a quality score');
      return;
    }

    submitReviewMutation.mutate({
      conceptId: currentReviewItem.concept_id,
      qualityScore: selectedQuality
    });
  };

  // Quality score descriptions
  const qualityDescriptions = [
    { value: 0, label: 'Complete blackout', description: "Couldn't recall anything" },
    { value: 1, label: 'Incorrect', description: 'Recalled incorrectly' },
    { value: 2, label: 'Incorrect but close', description: 'Almost correct but with mistakes' },
    { value: 3, label: 'Correct with difficulty', description: 'Recalled correctly but with effort' },
    { value: 4, label: 'Correct with hesitation', description: 'Recalled correctly after some thought' },
    { value: 5, label: 'Perfect recall', description: 'Recalled easily and confidently' }
  ];

  // Dashboard stats
  const dashboardStats = {
    todayStudyTime: todaySessionsData?.data?.todayStats?.total_time || 0,
    weeklyAverage: dashboardData?.data?.insights?.totalStudyTime ? 
      Math.round(dashboardData.data.insights.totalStudyTime / 7) : 0,
    currentStreak: progressData?.data?.profile?.current_streak_days || 0,
    reviewsDue: reviewData?.data?.count || 0,
    completedLessons: progressData?.data?.summary?.completedLessons || 0,
    masteredLessons: progressData?.data?.summary?.masteredLessons || 0
  };

  // Render spaced repetition widget
  const renderSpacedRepetitionWidget = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800 border border-gray-700 rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500 bg-opacity-20 rounded-lg">
            <Repeat className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Spaced Repetition</h3>
            <p className="text-sm text-gray-400">Reinforce your knowledge</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">{dashboardStats.reviewsDue}</div>
          <div className="text-sm text-gray-400">due today</div>
        </div>
      </div>

      {dashboardStats.reviewsDue > 0 ? (
        <>
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Review Progress</span>
              <span>{Math.max(0, reviewQueue.length - dashboardStats.reviewsDue)}/{reviewQueue.length}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ 
                  width: `${reviewQueue.length > 0 ? 
                    ((reviewQueue.length - dashboardStats.reviewsDue) / reviewQueue.length) * 100 
                    : 0}%` 
                }}
              />
            </div>
          </div>

          <div className="space-y-2 mb-4">
            {reviewQueue.slice(0, 3).map((item, index) => (
              <div key={item.concept_id} className="flex items-center justify-between p-2 bg-gray-700 rounded">
                <div>
                  <div className="text-sm font-medium text-white">{item.concept_title}</div>
                  <div className="text-xs text-gray-400 capitalize">{item.concept_type}</div>
                </div>
                <div className="text-xs text-gray-400">
                  {item.repetitions > 0 ? `${item.repetitions} reviews` : 'New'}
                </div>
              </div>
            ))}
            {reviewQueue.length > 3 && (
              <div className="text-center text-sm text-gray-400">
                +{reviewQueue.length - 3} more concepts
              </div>
            )}
          </div>

          <button
            onClick={startReviewSession}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4" />
            Start Review Session
          </button>
        </>
      ) : (
        <div className="text-center py-4">
          <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-2" />
          <p className="text-white font-medium">All caught up!</p>
          <p className="text-sm text-gray-400">No reviews due today</p>
        </div>
      )}
    </motion.div>
  );

  // Render review modal
  const renderReviewModal = () => {
    if (!showReviewModal || !currentReviewItem) return null;

    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-white">Review Session</h3>
                <p className="text-sm text-gray-400">
                  Concept {reviewIndex + 1} of {reviewQueue.length} • 
                  Difficulty: {currentReviewItem.easiness_factor?.toFixed(1) || '2.5'}
                </p>
              </div>
              <button
                onClick={() => {
                  setShowReviewModal(false);
                  resetReviewState();
                }}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Progress bar */}
            <div className="mb-6">
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((reviewIndex + 1) / reviewQueue.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Concept */}
            <div className="mb-6">
              <div className="bg-gray-700 rounded-lg p-4 mb-4">
                <div className="text-sm text-gray-400 mb-2 capitalize">
                  {currentReviewItem.concept_type} • {currentReviewItem.repetitions} previous reviews
                </div>
                <h4 className="text-lg font-semibold text-white mb-3">
                  {currentReviewItem.concept_title}
                </h4>
                <div className="text-white mb-4">
                  {currentReviewItem.question_text}
                </div>
                
                {/* Answer reveal */}
                {!showAnswer ? (
                  <button
                    onClick={() => setShowAnswer(true)}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    Show Answer
                  </button>
                ) : (
                  <div className="border-t border-gray-600 pt-4">
                    <div className="text-sm text-gray-400 mb-2">Answer:</div>
                    <div className="text-white mb-3">{currentReviewItem.answer_text}</div>
                    {currentReviewItem.hint_text && (
                      <div className="text-sm text-blue-400">
                        💡 Hint: {currentReviewItem.hint_text}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Quality selection */}
            {showAnswer && (
              <div className="mb-6">
                <h5 className="text-white font-medium mb-3">How well did you recall this concept?</h5>
                <div className="grid grid-cols-1 gap-2">
                  {qualityDescriptions.map((quality) => (
                    <button
                      key={quality.value}
                      onClick={() => setSelectedQuality(quality.value)}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        selectedQuality === quality.value
                          ? 'border-purple-500 bg-purple-500 bg-opacity-20'
                          : 'border-gray-600 hover:border-gray-500'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">{quality.label}</div>
                          <div className="text-sm text-gray-400">{quality.description}</div>
                        </div>
                        <div className="text-lg font-bold text-gray-400">
                          {quality.value}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            {showAnswer && (
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setShowReviewModal(false);
                    resetReviewState();
                  }}
                  className="flex-1 px-4 py-2 border border-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Skip Session
                </button>
                <button
                  onClick={handleReviewSubmit}
                  disabled={selectedQuality === null || submitReviewMutation.isLoading}
                  className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  {submitReviewMutation.isLoading ? 'Submitting...' : 'Submit & Continue'}
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      </AnimatePresence>
    );
  };

  // Render today's stats
  const renderTodayStats = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-blue-500 bg-opacity-20 rounded-lg">
            <Clock className="w-5 h-5 text-blue-400" />
          </div>
          <span className="text-gray-400">Today's Study Time</span>
        </div>
        <div className="text-2xl font-bold text-white mb-1">
          {Math.floor(dashboardStats.todayStudyTime / 60)}h {dashboardStats.todayStudyTime % 60}m
        </div>
        <div className="text-sm text-gray-400">
          Weekly avg: {Math.floor(dashboardStats.weeklyAverage / 60)}h {dashboardStats.weeklyAverage % 60}m
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-orange-500 bg-opacity-20 rounded-lg">
            <Flame className="w-5 h-5 text-orange-400" />
          </div>
          <span className="text-gray-400">Current Streak</span>
        </div>
        <div className="text-2xl font-bold text-white mb-1">
          {dashboardStats.currentStreak} days
        </div>
        <div className="text-sm text-gray-400">Keep it going!</div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-green-500 bg-opacity-20 rounded-lg">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <span className="text-gray-400">Lessons Completed</span>
        </div>
        <div className="text-2xl font-bold text-white mb-1">
          {dashboardStats.completedLessons}
        </div>
        <div className="text-sm text-gray-400">
          {dashboardStats.masteredLessons} mastered
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-purple-500 bg-opacity-20 rounded-lg">
            <Repeat className="w-5 h-5 text-purple-400" />
          </div>
          <span className="text-gray-400">Reviews Due</span>
        </div>
        <div className="text-2xl font-bold text-white mb-1">
          {dashboardStats.reviewsDue}
        </div>
        <div className="text-sm text-gray-400">
          {reviewStreak > 0 ? `${reviewStreak} streak` : 'Ready to review'}
        </div>
      </motion.div>
    </div>
  );

  if (isLoading || progressLoading) {
    return (
      <div className="dashboard h-full flex items-center justify-center">
        <LoadingSpinner
          size="large"
          text="Loading dashboard..."
          description="Preparing your learning insights"
        />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="dashboard min-h-screen bg-gray-900"
    >
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Neural Odyssey Dashboard</h1>
              <p className="text-gray-400 text-lg">
                Your personal machine learning journey companion
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => refetch()}
                className="p-2 bg-gray-800 border border-gray-600 hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
                title="Refresh Data"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* Today's Stats */}
        {renderTodayStats()}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Progress Map */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <ProgressMap />
            </motion.div>

            {/* Skill Tree */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <SkillTree />
            </motion.div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Spaced Repetition Widget */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7 }}
            >
              {renderSpacedRepetitionWidget()}
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 }}
              className="bg-gray-800 border border-gray-700 rounded-lg p-6"
            >
              <h3 className="font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={() => navigate('/learning')}
                  className="w-full flex items-center gap-3 p-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-white"
                >
                  <BookOpen className="w-5 h-5" />
                  <span>Continue Learning</span>
                  <ChevronRight className="w-4 h-4 ml-auto" />
                </button>
                <button
                  onClick={() => navigate('/quests')}
                  className="w-full flex items-center gap-3 p-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-white"
                >
                  <Target className="w-5 h-5" />
                  <span>Browse Quests</span>
                  <ChevronRight className="w-4 h-4 ml-auto" />
                </button>
                <button
                  onClick={() => navigate('/vault')}
                  className="w-full flex items-center gap-3 p-3 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white"
                >
                  <Sparkles className="w-5 h-5" />
                  <span>Explore Vault</span>
                  <ChevronRight className="w-4 h-4 ml-auto" />
                </button>
              </div>
            </motion.div>

            {/* Recent Vault Items */}
            {vaultPreviewData?.data && vaultPreviewData.data.length > 0 && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.9 }}
                className="bg-gray-800 border border-gray-700 rounded-lg p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-white">Recent Discoveries</h3>
                  <button
                    onClick={() => navigate('/vault')}
                    className="text-purple-400 hover:text-purple-300 text-sm"
                  >
                    View All
                  </button>
                </div>
                <div className="space-y-3">
                  {vaultPreviewData.data.slice(0, 3).map((item, index) => (
                    <div
                      key={item.itemId || index}
                      className="p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors cursor-pointer"
                      onClick={() => {
                        setSelectedVaultItem(item);
                        setShowVaultModal(true);
                      }}
                    >
                      <div className="text-sm font-medium text-white mb-1">
                        {item.title}
                      </div>
                      <div className="text-xs text-gray-400 capitalize">
                        {item.category?.replace('_', ' ')} • {item.rarity}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>

      {/* Review Modal */}
      {renderReviewModal()}

      {/* Vault Modal */}
      <VaultRevealModal
        vaultItem={selectedVaultItem}
        isOpen={showVaultModal}
        onClose={() => {
          setShowVaultModal(false);
          setSelectedVaultItem(null);
        }}
      />
    </motion.div>
  );
};

export default Dashboard;