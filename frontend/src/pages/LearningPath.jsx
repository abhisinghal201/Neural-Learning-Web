/**
 * Neural Odyssey Learning Path Component
 *
 * Main learning path interface providing a comprehensive overview of the entire
 * Machine Learning curriculum organized into 4 phases with interactive navigation,
 * progress tracking, and gamified learning elements.
 *
 * Features:
 * - Interactive phase and week navigation
 * - Real-time progress visualization
 * - Gamified learning path with unlocks and achievements
 * - Session type recommendations (math, coding, visual, applications)
 * - Adaptive content based on user progress
 * - Prerequisite checking and path validation
 * - Visual learning map with neural network styling
 * - Quick access to lessons, quests, and vault items
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Brain,
  BookOpen,
  Code,
  Eye,
  Target,
  Clock,
  Star,
  Award,
  ChevronRight,
  Play,
  Pause,
  CheckCircle,
  Circle,
  Lock,
  Unlock,
  TrendingUp,
  BarChart3,
  Map,
  Compass,
  Flame,
  Zap,
  Trophy,
  Calendar,
  Filter,
  Search,
  Grid,
  List,
  RefreshCw,
  ArrowRight,
  Lightbulb,
  Sparkles,
  Timer,
  Users,
  Globe,
  Bookmark,
  Download,
  Settings,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Plus,
  Minus,
  AlertCircle,
  Info
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const LearningPath = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // State management
  const [selectedPhase, setSelectedPhase] = useState(parseInt(searchParams.get('phase')) || 1);
  const [expandedPhases, setExpandedPhases] = useState(new Set([1]));
  const [viewMode, setViewMode] = useState(searchParams.get('view') || 'interactive');
  const [filterStatus, setFilterStatus] = useState('all');
  const [showOnlyUnlocked, setShowOnlyUnlocked] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSessionType, setSelectedSessionType] = useState(null);

  // Fetch learning progress data
  const { data: progressData, isLoading } = useQuery(
    'learningProgress',
    () => api.get('/learning/progress'),
    {
      refetchInterval: 30000,
      staleTime: 15000
    }
  );

  const { data: pathData } = useQuery(
    'learningPath',
    () => api.get('/learning/path'),
    {
      refetchInterval: 60000
    }
  );

  const { data: recommendationsData } = useQuery(
    'pathRecommendations',
    () => api.get('/learning/recommendations'),
    {
      refetchInterval: 120000
    }
  );

  // Learning path structure
  const learningPhases = useMemo(() => [
    {
      id: 1,
      title: 'Mathematical Foundations and Historical Context',
      subtitle: 'Building the Mathematical Foundation',
      description: 'Master the mathematical fundamentals that power all machine learning algorithms',
      duration: 'Months 1-3',
      color: 'from-blue-500 to-cyan-500',
      icon: Brain,
      totalWeeks: 12,
      focus: ['Linear Algebra', 'Calculus', 'Probability', 'Statistics'],
      outcomes: ['Mathematical intuition', 'Foundation skills', 'Historical perspective'],
      weeks: [
        { id: 1, title: 'Linear Algebra Through the Lens of Data', sessions: 4 },
        { id: 2, title: 'Calculus as the Engine of Learning', sessions: 4 },
        { id: 3, title: 'Probability and Statistics', sessions: 4 },
        { id: 4, title: 'The Birth of Machine Learning', sessions: 4 },
        { id: 5, title: 'Statistical Learning Theory', sessions: 4 },
        { id: 6, title: 'Information Theory Foundations', sessions: 4 },
        { id: 7, title: 'Optimization Theory', sessions: 4 },
        { id: 8, title: 'Linear Models Deep Dive', sessions: 4 },
        { id: 9, title: 'Bayesian Methods', sessions: 4 },
        { id: 10, title: 'Dimensionality and Complexity', sessions: 4 },
        { id: 11, title: 'Mathematical Synthesis', sessions: 4 },
        { id: 12, title: 'Phase 1 Integration Project', sessions: 4 }
      ]
    },
    {
      id: 2,
      title: 'Core Machine Learning with Deep Understanding',
      subtitle: 'Mastering Classical ML',
      description: 'Deep dive into supervised and unsupervised learning with hands-on implementation',
      duration: 'Months 4-6',
      color: 'from-green-500 to-emerald-500',
      icon: Target,
      totalWeeks: 12,
      focus: ['Supervised Learning', 'Unsupervised Learning', 'Neural Networks', 'Practical Applications'],
      outcomes: ['Algorithm mastery', 'Implementation skills', 'Real-world applications'],
      weeks: [
        { id: 13, title: 'The Learning Problem', sessions: 4 },
        { id: 14, title: 'Decision Trees and Ensemble Methods', sessions: 4 },
        { id: 15, title: 'Support Vector Machines', sessions: 4 },
        { id: 16, title: 'Pattern Discovery Without Labels', sessions: 4 },
        { id: 17, title: 'Neural Networks and Deep Learning', sessions: 4 },
        { id: 18, title: 'Advanced Optimization', sessions: 4 },
        { id: 19, title: 'Model Selection and Validation', sessions: 4 },
        { id: 20, title: 'Feature Engineering Mastery', sessions: 4 },
        { id: 21, title: 'Regularization Techniques', sessions: 4 },
        { id: 22, title: 'Business Applications', sessions: 4 },
        { id: 23, title: 'Algorithm Comparison Framework', sessions: 4 },
        { id: 24, title: 'Phase 2 Integration', sessions: 4 }
      ]
    },
    {
      id: 3,
      title: 'Advanced Topics and Modern AI',
      subtitle: 'The Transformer Revolution',
      description: 'Explore cutting-edge AI including transformers, attention, and modern architectures',
      duration: 'Months 7-9',
      color: 'from-purple-500 to-pink-500',
      icon: Zap,
      totalWeeks: 12,
      focus: ['Transformers', 'Attention Mechanisms', 'Modern Architectures', 'Advanced Applications'],
      outcomes: ['Modern AI understanding', 'Transformer mastery', 'Advanced implementations'],
      weeks: [
        { id: 25, title: 'The Attention Mechanism', sessions: 4 },
        { id: 26, title: 'Complete Transformer Implementation', sessions: 4 },
        { id: 27, title: 'Large Language Models', sessions: 4 },
        { id: 28, title: 'Computer Vision with Transformers', sessions: 4 },
        { id: 29, title: 'Multimodal AI Systems', sessions: 4 },
        { id: 30, title: 'Generative AI and Diffusion Models', sessions: 4 },
        { id: 31, title: 'Reinforcement Learning', sessions: 4 },
        { id: 32, title: 'Graph Neural Networks', sessions: 4 },
        { id: 33, title: 'Federated Learning', sessions: 4 },
        { id: 34, title: 'AI Safety and Alignment', sessions: 4 },
        { id: 35, title: 'Research Frontiers', sessions: 4 },
        { id: 36, title: 'Phase 3 Capstone', sessions: 4 }
      ]
    },
    {
      id: 4,
      title: 'Mastery and Innovation',
      subtitle: 'Research and Real-World Impact',
      description: 'Apply your knowledge to research problems and create innovative solutions',
      duration: 'Months 10-12',
      color: 'from-orange-500 to-red-500',
      icon: Trophy,
      totalWeeks: 12,
      focus: ['Research Methods', 'Innovation Projects', 'Industry Applications', 'Leadership'],
      outcomes: ['Research capability', 'Innovation skills', 'Industry readiness'],
      weeks: [
        { id: 37, title: 'Research Methodology', sessions: 4 },
        { id: 38, title: 'Paper Implementation Challenge', sessions: 4 },
        { id: 39, title: 'Original Research Project', sessions: 4 },
        { id: 40, title: 'Industry Case Studies', sessions: 4 },
        { id: 41, title: 'Startup and Entrepreneurship', sessions: 4 },
        { id: 42, title: 'Technical Leadership', sessions: 4 },
        { id: 43, title: 'Open Source Contribution', sessions: 4 },
        { id: 44, title: 'Conference Presentation', sessions: 4 },
        { id: 45, title: 'Portfolio Development', sessions: 4 },
        { id: 46, title: 'Capstone Project', sessions: 4 },
        { id: 47, title: 'Knowledge Transfer', sessions: 4 },
        { id: 48, title: 'Neural Odyssey Completion', sessions: 4 }
      ]
    }
  ], []);

  // Session types configuration
  const sessionTypes = [
    {
      id: 'math',
      label: 'Mathematical Deep Dive',
      icon: Brain,
      color: 'blue',
      description: 'Theory, proofs, and mathematical intuition'
    },
    {
      id: 'coding',
      label: 'Coding Practice',
      icon: Code,
      color: 'green',
      description: 'Implementation and algorithm practice'
    },
    {
      id: 'visual',
      label: 'Visual Projects',
      icon: Eye,
      color: 'purple',
      description: 'Visualization and interactive demos'
    },
    {
      id: 'applications',
      label: 'Real Applications',
      icon: Target,
      color: 'orange',
      description: 'Business cases and practical applications'
    }
  ];

  // Calculate progress statistics
  const progressStats = useMemo(() => {
    if (!progressData?.data) return null;

    const progress = progressData.data.progress || [];
    const profile = progressData.data.profile || {};

    const totalLessons = learningPhases.reduce((sum, phase) => sum + phase.totalWeeks * 4, 0);
    const completedLessons = progress.filter(p => p.status === 'completed' || p.status === 'mastered').length;
    const masteredLessons = progress.filter(p => p.status === 'mastered').length;

    const currentPhase = profile.current_phase || 1;
    const currentWeek = profile.current_week || 1;

    return {
      totalLessons,
      completedLessons,
      masteredLessons,
      overallProgress: (completedLessons / totalLessons) * 100,
      currentPhase,
      currentWeek,
      totalStudyTime: profile.total_study_minutes || 0,
      currentStreak: profile.current_streak_days || 0
    };
  }, [progressData, learningPhases]);

  // Get week progress for a specific phase
  const getWeekProgress = (phaseId, weekId) => {
    if (!progressData?.data?.progress) return { completed: 0, total: 4, status: 'locked' };

    const weekProgress = progressData.data.progress.filter(
      p => p.phase === phaseId && p.week === weekId
    );

    const completed = weekProgress.filter(p => p.status === 'completed' || p.status === 'mastered').length;
    const total = 4; // 4 sessions per week

    let status = 'locked';
    if (phaseId < (progressStats?.currentPhase || 1)) {
      status = 'available';
    } else if (phaseId === (progressStats?.currentPhase || 1)) {
      if (weekId <= (progressStats?.currentWeek || 1)) {
        status = 'available';
      }
    }

    if (completed === total) status = 'completed';
    if (completed > 0 && completed < total) status = 'in_progress';

    return { completed, total, status, progress: (completed / total) * 100 };
  };

  // Handle phase expansion
  const togglePhaseExpansion = (phaseId) => {
    const newExpanded = new Set(expandedPhases);
    if (newExpanded.has(phaseId)) {
      newExpanded.delete(phaseId);
    } else {
      newExpanded.add(phaseId);
    }
    setExpandedPhases(newExpanded);
  };

  // Handle navigation to specific week
  const navigateToWeek = (phase, week) => {
    const weekProgress = getWeekProgress(phase, week);
    if (weekProgress.status === 'locked') {
      toast.error('Complete previous weeks to unlock this content');
      return;
    }
    navigate(`/learning/phase/${phase}/week/${week}`);
  };

  // Handle starting a learning session
  const startSession = (sessionType, phase, week) => {
    navigate(`/learning/phase/${phase}/week/${week}?session=${sessionType}`);
  };

  // Format time helper
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className="learning-path h-full">
        <LoadingSpinner
          size="large"
          text="Loading learning path..."
          description="Analyzing your progress and preparing personalized content"
        />
      </div>
    );
  }

  // Render phase card
  const renderPhaseCard = (phase) => {
    const isExpanded = expandedPhases.has(phase.id);
    const isCurrentPhase = phase.id === progressStats?.currentPhase;
    const isUnlocked = phase.id <= (progressStats?.currentPhase || 1);

    // Calculate phase progress
    const phaseProgress = phase.weeks.map(week => getWeekProgress(phase.id, week.id));
    const completedWeeks = phaseProgress.filter(w => w.status === 'completed').length;
    const totalWeeks = phase.weeks.length;
    const phaseCompletionPercent = (completedWeeks / totalWeeks) * 100;

    return (
      <motion.div
        key={phase.id}
        layout
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`
          bg-gray-800 border border-gray-700 rounded-xl overflow-hidden
          ${isCurrentPhase ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
          ${!isUnlocked ? 'opacity-60' : ''}
        `}
      >
        {/* Phase Header */}
        <div
          className={`
            p-6 bg-gradient-to-r ${phase.color} cursor-pointer
            ${!isUnlocked ? 'cursor-not-allowed' : ''}
          `}
          onClick={() => isUnlocked && togglePhaseExpansion(phase.id)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                <phase.icon className="w-6 h-6 text-white" />
              </div>
              
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="text-xl font-bold text-white">
                    Phase {phase.id}: {phase.title}
                  </h3>
                  {!isUnlocked && <Lock className="w-4 h-4 text-white opacity-70" />}
                  {isCurrentPhase && (
                    <span className="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs text-white font-medium">
                      Current
                    </span>
                  )}
                </div>
                <p className="text-white text-opacity-90 text-sm">{phase.subtitle}</p>
                <p className="text-white text-opacity-75 text-xs">{phase.duration}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Progress Ring */}
              <div className="relative w-16 h-16">
                <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                  <circle
                    cx="32"
                    cy="32"
                    r="28"
                    stroke="white"
                    strokeOpacity="0.2"
                    strokeWidth="4"
                    fill="none"
                  />
                  <circle
                    cx="32"
                    cy="32"
                    r="28"
                    stroke="white"
                    strokeWidth="4"
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={`${phaseCompletionPercent * 1.76} 176`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-white font-bold text-sm">
                    {Math.round(phaseCompletionPercent)}%
                  </span>
                </div>
              </div>

              {isUnlocked && (
                <ChevronDown
                  className={`w-5 h-5 text-white transition-transform duration-200 ${
                    isExpanded ? 'transform rotate-180' : ''
                  }`}
                />
              )}
            </div>
          </div>

          {/* Phase Description */}
          <div className="mt-4">
            <p className="text-white text-opacity-90 text-sm mb-3">{phase.description}</p>
            
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="bg-white bg-opacity-10 rounded-lg p-2">
                <div className="text-xs text-white text-opacity-75">Focus Areas</div>
                <div className="text-sm text-white font-medium">{phase.focus.length} Topics</div>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-2">
                <div className="text-xs text-white text-opacity-75">Duration</div>
                <div className="text-sm text-white font-medium">{totalWeeks} Weeks</div>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-2">
                <div className="text-xs text-white text-opacity-75">Progress</div>
                <div className="text-sm text-white font-medium">{completedWeeks}/{totalWeeks}</div>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-2">
                <div className="text-xs text-white text-opacity-75">Status</div>
                <div className="text-sm text-white font-medium">
                  {!isUnlocked ? 'Locked' : isCurrentPhase ? 'Active' : completedWeeks === totalWeeks ? 'Completed' : 'Available'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Phase Content */}
        <AnimatePresence>
          {isExpanded && isUnlocked && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="p-6 bg-gray-800">
                {/* Focus Areas */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-white mb-3">Focus Areas</h4>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                    {phase.focus.map((focus, index) => (
                      <div
                        key={index}
                        className="bg-gray-700 rounded-lg px-3 py-2 text-center"
                      >
                        <span className="text-sm text-gray-300">{focus}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Learning Outcomes */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-white mb-3">Learning Outcomes</h4>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
                    {phase.outcomes.map((outcome, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-sm text-gray-300">{outcome}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Week Grid */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-white mb-3">Weekly Breakdown</h4>
                  <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                    {phase.weeks.map((week) => {
                      const weekProgress = getWeekProgress(phase.id, week.id);
                      const isCurrentWeek = phase.id === progressStats?.currentPhase && week.id === progressStats?.currentWeek;

                      return (
                        <motion.div
                          key={week.id}
                          whileHover={weekProgress.status !== 'locked' ? { scale: 1.02 } : {}}
                          className={`
                            bg-gray-700 border border-gray-600 rounded-lg p-4 cursor-pointer
                            transition-all duration-200
                            ${weekProgress.status === 'locked' ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-500'}
                            ${isCurrentWeek ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
                          `}
                          onClick={() => navigateToWeek(phase.id, week.id)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-sm font-medium text-white">Week {week.id}</span>
                              {weekProgress.status === 'locked' && <Lock className="w-3 h-3 text-gray-400" />}
                              {weekProgress.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-400" />}
                              {isCurrentWeek && <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />}
                            </div>
                            <span className="text-xs text-gray-400">{week.sessions} sessions</span>
                          </div>

                          <h5 className="text-sm font-medium text-white mb-2 line-clamp-2">
                            {week.title}
                          </h5>

                          {/* Progress Bar */}
                          <div className="w-full bg-gray-600 rounded-full h-2 mb-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-300 ${
                                weekProgress.status === 'completed' ? 'bg-green-400' :
                                weekProgress.status === 'in_progress' ? 'bg-blue-400' :
                                'bg-gray-600'
                              }`}
                              style={{ width: `${weekProgress.progress}%` }}
                            />
                          </div>

                          <div className="flex items-center justify-between text-xs text-gray-400">
                            <span>{weekProgress.completed}/{weekProgress.total} completed</span>
                            <span className="capitalize">{weekProgress.status}</span>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>

                {/* Quick Actions */}
                {isCurrentPhase && (
                  <div className="border-t border-gray-700 pt-6">
                    <h4 className="text-lg font-semibold text-white mb-3">Quick Actions</h4>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      {sessionTypes.map((sessionType) => (
                        <button
                          key={sessionType.id}
                          onClick={() => startSession(sessionType.id, phase.id, progressStats?.currentWeek || 1)}
                          className={`
                            p-4 bg-${sessionType.color}-600 hover:bg-${sessionType.color}-700
                            rounded-lg transition-colors text-white
                            flex flex-col items-center space-y-2
                          `}
                        >
                          <sessionType.icon className="w-6 h-6" />
                          <span className="text-sm font-medium text-center">{sessionType.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="learning-path min-h-screen bg-gray-900"
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
              <h1 className="text-4xl font-bold text-white mb-2">Neural Learning Path</h1>
              <p className="text-gray-400 text-lg">
                Your comprehensive journey to Machine Learning mastery
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2">
                <div className="text-xs text-gray-400">Overall Progress</div>
                <div className="text-2xl font-bold text-white">
                  {Math.round(progressStats?.overallProgress || 0)}%
                </div>
              </div>
              
              <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2">
                <div className="text-xs text-gray-400">Study Time</div>
                <div className="text-2xl font-bold text-white">
                  {formatTime(progressStats?.totalStudyTime || 0)}
                </div>
              </div>

              <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2">
                <div className="text-xs text-gray-400">Current Streak</div>
                <div className="text-2xl font-bold text-white flex items-center space-x-1">
                  <Flame className="w-5 h-5 text-orange-400" />
                  <span>{progressStats?.currentStreak || 0}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Progress Overview */}
          {progressStats && (
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Learning Progress Overview</h3>
                <div className="text-sm text-gray-400">
                  Phase {progressStats.currentPhase} • Week {progressStats.currentWeek}
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">{progressStats.completedLessons}</div>
                  <div className="text-sm text-gray-400">Lessons Completed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">{progressStats.masteredLessons}</div>
                  <div className="text-sm text-gray-400">Lessons Mastered</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">{progressStats.totalLessons}</div>
                  <div className="text-sm text-gray-400">Total Lessons</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-400">{Math.round(progressStats.overallProgress)}%</div>
                  <div className="text-sm text-gray-400">Overall Progress</div>
                </div>
              </div>

              {/* Overall Progress Bar */}
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-3">
                  <div
                    className="h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                    style={{ width: `${progressStats.overallProgress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {recommendationsData?.data?.nextActions && (
            <div className="bg-gradient-to-r from-blue-900 to-purple-900 border border-blue-700 rounded-lg p-4 mb-6">
              <div className="flex items-center space-x-2 mb-2">
                <Lightbulb className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">Recommended Next Actions</h3>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {recommendationsData.data.nextActions.slice(0, 3).map((action, index) => (
                  <div key={index} className="bg-white bg-opacity-10 rounded-lg p-3">
                    <div className="text-sm font-medium text-white">{action.title}</div>
                    <div className="text-xs text-gray-300 mt-1">{action.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Learning Phases */}
        <div className="space-y-6">
          {learningPhases.map((phase) => renderPhaseCard(phase))}
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center text-gray-400"
        >
          <p>Your Neural Odyssey awaits. Every lesson brings you closer to mastery.</p>
        </div>
      </div>
    </motion.div>
  );
};

export default LearningPath;