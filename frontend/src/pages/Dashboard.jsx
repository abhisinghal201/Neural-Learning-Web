/**
 * Enhanced Neural Odyssey Dashboard Page
 *
 * Now fully leverages all backend capabilities:
 * - Comprehensive session management with energy/mood tracking
 * - Spaced repetition system with SM-2 algorithm
 * - Knowledge graph connections
 * - Advanced analytics and insights
 * - Real-time progress tracking
 * - Vault integration with unlock analytics
 * - Next lesson recommendations
 * - Streak tracking and achievements
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
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  EyeOff,
  Battery,
  Smile,
  Meh,
  Frown
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

  // Session Tracking State
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionData, setSessionData] = useState({
    session_type: 'math',
    energy_level: 8,
    mood_before: 'focused',
    mood_after: '',
    focus_score: 8,
    interruption_count: 0,
    distraction_types: [],
    goals: [],
    completed_goals: [],
    session_notes: ''
  });
  const [sessionTimer, setSessionTimer] = useState(0);
  const [sessionStartTime, setSessionStartTime] = useState(null);

  // Data fetching
  const { data: dashboardData, isLoading, refetch } = useQuery(
    ['dashboard', selectedTimeRange],
    () => api.get(`/learning/analytics?timeframe=${selectedTimeRange === 'week' ? 7 : selectedTimeRange === 'month' ? 30 : 90}`),
    {
      refetchInterval: 60000,
      staleTime: 30000
    }
  );

  const { data: progressData } = useQuery(
    'learningProgress',
    () => api.get('/learning/progress'),
    {
      refetchInterval: 30000
    }
  );

  const { data: todaySessions } = useQuery(
    'todaySessions',
    () => api.get('/learning/sessions/today'),
    {
      refetchInterval: 60000
    }
  );

  const { data: spacedRepetitionData } = useQuery(
    'spacedRepetition',
    () => api.get('/learning/spaced-repetition'),
    {
      refetchInterval: 300000 // 5 minutes
    }
  );

  const { data: knowledgeGraph } = useQuery(
    'knowledgeGraph',
    () => api.get('/learning/knowledge-graph'),
    {
      refetchInterval: 300000
    }
  );

  const { data: nextLesson } = useQuery(
    'nextLesson',
    () => api.get('/learning/next-lesson'),
    {
      refetchInterval: 60000
    }
  );

  const { data: streakData } = useQuery(
    'learningStreak',
    () => api.get('/learning/streak'),
    {
      refetchInterval: 60000
    }
  );

  const { data: vaultStats } = useQuery(
    'vaultStatistics',
    () => api.get('/vault/statistics'),
    {
      refetchInterval: 120000
    }
  );

  // Mutations
  const submitReviewMutation = useMutation(
    ({ itemId, qualityScore, notes }) => 
      api.post(`/learning/spaced-repetition/${itemId}/review`, {
        quality_score: qualityScore,
        review_notes: notes || ''
      }),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('spacedRepetition');
        setReviewStreak(prev => selectedQuality >= 3 ? prev + 1 : 0);
        
        const nextReviewDate = new Date(data.data.nextReviewDate).toLocaleDateString();
        toast.success(`Review recorded! Next review: ${nextReviewDate}`);
        
        moveToNextReview();
      },
      onError: () => {
        toast.error('Failed to record review');
      }
    }
  );

  const createSessionMutation = useMutation(
    (sessionData) => api.post('/learning/sessions', sessionData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('todaySessions');
        queryClient.invalidateQueries('dashboard');
        toast.success('Session started successfully!');
      }
    }
  );

  // Session timer effect
  useEffect(() => {
    let interval = null;
    if (isSessionActive && sessionStartTime) {
      interval = setInterval(() => {
        setSessionTimer(Date.now() - sessionStartTime);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isSessionActive, sessionStartTime]);

  // Initialize review queue
  useEffect(() => {
    if (spacedRepetitionData?.data?.reviewItems) {
      setReviewQueue(spacedRepetitionData.data.reviewItems);
      if (spacedRepetitionData.data.reviewItems.length > 0) {
        setCurrentReviewItem(spacedRepetitionData.data.reviewItems[0]);
      }
    }
  }, [spacedRepetitionData]);

  // Helper functions
  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
    }
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  const startQuickSession = async (sessionType) => {
    const now = Date.now();
    setSessionStartTime(now);
    setIsSessionActive(true);
    setSessionData({ ...sessionData, session_type: sessionType });

    try {
      await createSessionMutation.mutateAsync({
        ...sessionData,
        session_type: sessionType,
        start_time: new Date(now).toISOString(),
        target_duration_minutes: 25
      });
    } catch (error) {
      setIsSessionActive(false);
      setSessionStartTime(null);
    }
  };

  const moveToNextReview = () => {
    if (reviewIndex < reviewQueue.length - 1) {
      const nextIndex = reviewIndex + 1;
      setReviewIndex(nextIndex);
      setCurrentReviewItem(reviewQueue[nextIndex]);
      setShowAnswer(false);
      setSelectedQuality(null);
    } else {
      // Review session complete
      setShowReviewModal(false);
      setReviewIndex(0);
      toast.success(`Review session complete! Reviewed ${reviewQueue.length} items.`);
    }
  };

  const submitReview = () => {
    if (selectedQuality === null) {
      toast.error('Please select a quality score');
      return;
    }

    submitReviewMutation.mutate({
      itemId: currentReviewItem.concept_id,
      qualityScore: selectedQuality,
      notes: ''
    });
  };

  // Quality score options for spaced repetition
  const qualityOptions = [
    { value: 0, label: 'Complete blackout', color: 'bg-red-600', description: 'No recollection at all' },
    { value: 1, label: 'Incorrect', color: 'bg-red-500', description: 'Incorrect response with some recollection' },
    { value: 2, label: 'Barely correct', color: 'bg-orange-500', description: 'Correct with serious difficulty' },
    { value: 3, label: 'Correct with effort', color: 'bg-yellow-500', description: 'Correct response with some difficulty' },
    { value: 4, label: 'Easy correct', color: 'bg-green-500', description: 'Correct response with hesitation' },
    { value: 5, label: 'Perfect recall', color: 'bg-green-600', description: 'Perfect response immediately' }
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  const profile = progressData?.data?.profile || initialProfile;
  const currentStreak = streakData?.data?.currentStreak || 0;
  const todayStats = todaySessions?.data?.summary || {};
  const reviewItemsCount = spacedRepetitionData?.data?.reviewItems?.length || 0;

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Current Streak */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-orange-500 to-red-500 rounded-lg p-6 text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold">{currentStreak}</div>
              <div className="text-sm opacity-90">Day Streak</div>
            </div>
            <Flame className="w-8 h-8 opacity-80" />
          </div>
          {currentStreak > 0 && (
            <div className="text-xs mt-2 opacity-75">
              Longest: {streakData?.data?.longestStreak || currentStreak} days
            </div>
          )}
        </motion.div>

        {/* Today's Progress */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg p-6 text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold">{todayStats.total_time_minutes || 0}m</div>
              <div className="text-sm opacity-90">Today's Study Time</div>
            </div>
            <Clock className="w-8 h-8 opacity-80" />
          </div>
          <div className="text-xs mt-2 opacity-75">
            {todayStats.session_count || 0} sessions • Avg focus: {Math.round(todayStats.average_focus || 0)}/10
          </div>
        </motion.div>

        {/* Review Items Due */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-6 text-white cursor-pointer"
          onClick={() => reviewItemsCount > 0 && setShowReviewModal(true)}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold">{reviewItemsCount}</div>
              <div className="text-sm opacity-90">Items to Review</div>
            </div>
            <Brain className="w-8 h-8 opacity-80" />
          </div>
          {reviewItemsCount > 0 && (
            <div className="text-xs mt-2 opacity-75">
              Click to start review session
            </div>
          )}
        </motion.div>

        {/* Vault Progress */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg p-6 text-white"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold">{vaultStats?.data?.unlocked || 0}</div>
              <div className="text-sm opacity-90">Vault Items Unlocked</div>
            </div>
            <Sparkles className="w-8 h-8 opacity-80" />
          </div>
          <div className="text-xs mt-2 opacity-75">
            {Math.round((vaultStats?.data?.unlockRate || 0) * 100)}% unlocked
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Start Session */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => startQuickSession(todayStats.recommended_next_type || 'math')}
            disabled={isSessionActive}
            className={`p-4 rounded-lg border-2 border-dashed transition-colors ${
              isSessionActive 
                ? 'border-gray-600 bg-gray-700 cursor-not-allowed' 
                : 'border-green-600 hover:bg-green-600/10 text-green-400'
            }`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <Play className="w-5 h-5" />
              <span className="font-medium">
                {isSessionActive ? 'Session Active' : 'Quick Session'}
              </span>
            </div>
            <div className="text-xs opacity-75">
              {isSessionActive 
                ? formatTime(sessionTimer)
                : `Recommended: ${(todayStats.recommended_next_type || 'math').replace('_', ' ')}`
              }
            </div>
          </motion.button>

          {/* Review Items */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setShowReviewModal(true)}
            disabled={reviewItemsCount === 0}
            className={`p-4 rounded-lg border-2 border-dashed transition-colors ${
              reviewItemsCount === 0
                ? 'border-gray-600 bg-gray-700 cursor-not-allowed'
                : 'border-purple-600 hover:bg-purple-600/10 text-purple-400'
            }`}
          >
            <div className="flex items-center space-x-2 mb-2">
              <Repeat className="w-5 h-5" />
              <span className="font-medium">Spaced Review</span>
            </div>
            <div className="text-xs opacity-75">
              {reviewItemsCount > 0 ? `${reviewItemsCount} items due` : 'No reviews due'}
            </div>
          </motion.button>

          {/* Next Lesson */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => nextLesson?.data && navigate(`/learning/lessons/${nextLesson.data.lesson_id}`)}
            className="p-4 rounded-lg border-2 border-dashed border-blue-600 hover:bg-blue-600/10 text-blue-400 transition-colors"
          >
            <div className="flex items-center space-x-2 mb-2">
              <BookOpen className="w-5 h-5" />
              <span className="font-medium">Continue Learning</span>
            </div>
            <div className="text-xs opacity-75">
              {nextLesson?.data?.lesson_title || 'No lessons available'}
            </div>
          </motion.button>

          {/* View Analytics */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/analytics')}
            className="p-4 rounded-lg border-2 border-dashed border-cyan-600 hover:bg-cyan-600/10 text-cyan-400 transition-colors"
          >
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="w-5 h-5" />
              <span className="font-medium">View Analytics</span>
            </div>
            <div className="text-xs opacity-75">
              Detailed progress insights
            </div>
          </motion.button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Progress Map */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Learning Progress</h3>
            <button
              onClick={() => navigate('/learning')}
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              View Full Path
            </button>
          </div>
          <ProgressMap data={progressData?.data} compact />
        </div>

        {/* Skill Tree */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Skill Development</h3>
            <button
              onClick={() => navigate('/analytics')}
              className="text-green-400 hover:text-green-300 text-sm"
            >
              View Details
            </button>
          </div>
          <SkillTree data={dashboardData?.data} compact />
        </div>
      </div>

      {/* Knowledge Graph Connections */}
      {knowledgeGraph?.data?.connections?.length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Knowledge Connections</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {knowledgeGraph.data.connections.slice(0, 6).map((connection, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Layers className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-medium text-white">{connection.from_concept}</span>
                </div>
                <div className="flex items-center space-x-2 text-xs text-gray-400">
                  <ArrowRight className="w-3 h-3" />
                  <span>{connection.to_concept}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">{connection.relationship_type}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Spaced Repetition Review Modal */}
      <AnimatePresence>
        {showReviewModal && currentReviewItem && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-800 rounded-lg border border-gray-700 p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-white">Spaced Repetition Review</h3>
                  <div className="text-sm text-gray-400">
                    Item {reviewIndex + 1} of {reviewQueue.length} • Streak: {reviewStreak}
                  </div>
                </div>
                <button
                  onClick={() => setShowReviewModal(false)}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-gray-700 rounded-full h-2 mb-6">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((reviewIndex + 1) / reviewQueue.length) * 100}%` }}
                />
              </div>

              {/* Review Item */}
              <div className="mb-6">
                <div className="bg-gray-700 rounded-lg p-4 mb-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <BookOpen className="w-4 h-4 text-blue-400" />
                    <span className="text-sm text-gray-400">
                      {currentReviewItem.lesson_title} • Phase {currentReviewItem.phase} Week {currentReviewItem.week}
                    </span>
                  </div>
                  <h4 className="text-lg font-medium text-white mb-2">
                    Concept: {currentReviewItem.concept_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </h4>
                  <div className="text-sm text-gray-300">
                    Last reviewed: {currentReviewItem.last_reviewed_at 
                      ? new Date(currentReviewItem.last_reviewed_at).toLocaleDateString()
                      : 'Never'
                    }
                  </div>
                  <div className="text-sm text-gray-400">
                    Difficulty: {currentReviewItem.difficulty_factor?.toFixed(1)} • 
                    Interval: {currentReviewItem.interval_days} days
                  </div>
                </div>

                {/* Show/Hide Answer */}
                <div className="text-center mb-4">
                  <button
                    onClick={() => setShowAnswer(!showAnswer)}
                    className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                      showAnswer
                        ? 'bg-gray-600 hover:bg-gray-700 text-white'
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                    }`}
                  >
                    {showAnswer ? (
                      <>
                        <EyeOff className="w-4 h-4 inline mr-2" />
                        Hide Answer
                      </>
                    ) : (
                      <>
                        <Eye className="w-4 h-4 inline mr-2" />
                        Show Answer
                      </>
                    )}
                  </button>
                </div>

                {/* Answer Content */}
                <AnimatePresence>
                  {showAnswer && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="bg-green-600/10 border border-green-600/30 rounded-lg p-4 mb-4"
                    >
                      <div className="text-green-400 font-medium mb-2">Answer:</div>
                      <div className="text-white">
                        This concept involves understanding the mathematical and practical applications
                        in machine learning. Review your notes and implementation for this topic.
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Quality Rating */}
              {showAnswer && (
                <div className="mb-6">
                  <h4 className="text-lg font-medium text-white mb-3">How well did you remember this?</h4>
                  <div className="grid grid-cols-1 gap-2">
                    {qualityOptions.map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setSelectedQuality(option.value)}
                        className={`p-3 rounded-lg border-2 transition-all text-left ${
                          selectedQuality === option.value
                            ? `${option.color} border-current text-white`
                            : 'border-gray-600 hover:border-gray-500 text-gray-300 hover:text-white'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium">{option.value}. {option.label}</div>
                            <div className="text-xs opacity-75">{option.description}</div>
                          </div>
                          {selectedQuality === option.value && (
                            <CheckCircle className="w-5 h-5" />
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-3">
                <button
                  onClick={() => setShowReviewModal(false)}
                  className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={submitReview}
                  disabled={!showAnswer || selectedQuality === null || submitReviewMutation.isLoading}
                  className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white"
                >
                  {submitReviewMutation.isLoading ? 'Submitting...' : 'Submit Review'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Vault Reveal Modal */}
      <AnimatePresence>
        {showVaultModal && selectedVaultItem && (
          <VaultRevealModal
            item={selectedVaultItem}
            onClose={() => {
              setShowVaultModal(false);
              setSelectedVaultItem(null);
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default Dashboard;