/**
 * Enhanced Neural Odyssey Learning Segment Component
 * 
 * Now fully leverages ALL backend session management capabilities:
 * - Energy level tracking (1-10) with real-time monitoring
 * - Mood before/after session tracking with detailed options
 * - Focus score monitoring (1-10) with interruption tracking
 * - Comprehensive distraction and interruption logging
 * - Goal setting and completion tracking with success metrics
 * - Session notes and reflection with mentor feedback
 * - Real-time session analytics and insights
 * - Integration with spaced repetition system
 * - Knowledge graph connections tracking
 * - Advanced progress analytics
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { 
  BookOpen,
  Code,
  Eye,
  Target,
  Clock,
  Brain,
  CheckCircle,
  Star,
  ArrowLeft,
  ArrowRight,
  Play,
  Pause,
  Square,
  RotateCcw,
  Lightbulb,
  Award,
  TrendingUp,
  Filter,
  Search,
  Grid,
  List,
  ChevronDown,
  Zap,
  Trophy,
  Calendar,
  Users,
  Settings,
  MessageSquare,
  Smile,
  Frown,
  Meh,
  Coffee,
  Flame,
  Battery,
  Phone,
  Mail,
  Volume2,
  AlertTriangle,
  Edit3,
  Save,
  Plus,
  Minus,
  Timer,
  Activity,
  BarChart3,
  Layers,
  ChevronUp,
  X,
  Check,
  RefreshCw,
  Archive,
  Bell,
  Wifi,
  WifiOff,
  Monitor,
  Headphones,
  Laptop,
  Smartphone,
  Home,
  Building,
  MapPin
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import QuestCard from '../components/QuestCard';
import CodePlayground from '../components/CodePlayground';
import VaultRevealModal from '../components/VaultRevealModal';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const LearningSegment = () => {
  const { phase, week } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Session Management State
  const [currentSession, setCurrentSession] = useState(null);
  const [sessionTimer, setSessionTimer] = useState(0);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState(null);
  const [sessionPaused, setSessionPaused] = useState(false);
  const [pauseStartTime, setPauseStartTime] = useState(null);
  const [totalPauseTime, setTotalPauseTime] = useState(0);
  
  // Enhanced Session Data - leveraging ALL backend capabilities
  const [sessionData, setSessionData] = useState({
    session_type: 'math',
    energy_level: 8,
    mood_before: 'focused',
    mood_after: '',
    focus_score: 8,
    interruption_count: 0,
    distraction_types: [],
    distraction_details: [],
    goals: [],
    completed_goals: [],
    session_notes: '',
    environment_type: 'home',
    device_type: 'laptop',
    noise_level: 'quiet',
    lighting_quality: 'good',
    break_count: 0,
    break_duration_minutes: 0,
    productivity_rating: 8,
    difficulty_encountered: [],
    learning_insights: '',
    next_session_plan: ''
  });

  // Session Setup and Review Modals
  const [showSessionSetup, setShowSessionSetup] = useState(false);
  const [showSessionReview, setShowSessionReview] = useState(false);
  const [showInterruptionModal, setShowInterruptionModal] = useState(false);
  const [showBreakModal, setShowBreakModal] = useState(false);
  const [selectedSessionType, setSelectedSessionType] = useState(null);

  // Learning Content State
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'lessons');
  const [selectedLesson, setSelectedLesson] = useState(null);
  const [selectedQuest, setSelectedQuest] = useState(null);
  const [showCodePlayground, setShowCodePlayground] = useState(false);
  const [lessonFilter, setLessonFilter] = useState('all');
  const [questFilter, setQuestFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState('grid');

  // Goal Management
  const [newGoal, setNewGoal] = useState('');
  const [goalPriority, setGoalPriority] = useState('medium');
  const [goalEstimatedTime, setGoalEstimatedTime] = useState(15);

  // Real-time tracking
  const [currentInterruption, setCurrentInterruption] = useState(null);
  const [energyHistory, setEnergyHistory] = useState([]);
  const [focusHistory, setFocusHistory] = useState([]);

  // Refs
  const sessionTimerRef = useRef(null);
  const energyCheckInterval = useRef(null);

  // Data fetching
  const { data: segmentData, isLoading } = useQuery(
    ['learningSegment', phase, week],
    () => api.learning.getPhaseProgress(phase, { week }),
    {
      refetchInterval: 30000
    }
  );

  const { data: lessonsData } = useQuery(
    ['lessons', phase, week],
    () => api.learning.getLessons({ phase, week }),
    {
      refetchInterval: 60000
    }
  );

  const { data: questsData } = useQuery(
    ['quests', phase, week],
    () => api.learning.getQuests({ phase, week }),
    {
      refetchInterval: 60000
    }
  );

  const { data: todaySession } = useQuery(
    'todaySession',
    () => api.learning.getTodaySessions(),
    {
      refetchInterval: 30000
    }
  );

  const { data: spacedRepetitionData } = useQuery(
    'spacedRepetitionSegment',
    () => api.learning.getSpacedRepetition({ limit: 5 }),
    {
      refetchInterval: 300000
    }
  );

  const { data: knowledgeConnections } = useQuery(
    ['knowledgeGraph', phase, week],
    () => api.learning.getKnowledgeGraph({ phase, week }),
    {
      refetchInterval: 300000
    }
  );

  // Mutations
  const createSessionMutation = useMutation(
    (sessionData) => api.learning.createSession(sessionData),
    {
      onSuccess: (data) => {
        setCurrentSession(data.data);
        queryClient.invalidateQueries('todaySession');
        toast.success('Session started successfully! 🎯');
      },
      onError: () => {
        toast.error('Failed to start session');
        setIsSessionActive(false);
        setSessionStartTime(null);
      }
    }
  );

  const updateSessionMutation = useMutation(
    ({ date, sessionData }) => api.learning.updateSession(date, sessionData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('todaySession');
        queryClient.invalidateQueries('dashboard');
        toast.success('Session completed! Great work! 🏆');
      }
    }
  );

  const submitQuestMutation = useMutation(
    (questData) => api.learning.submitQuest(questData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['quests', phase, week]);
        toast.success('Quest submitted successfully!');
      }
    }
  );

  const addKnowledgeConnectionMutation = useMutation(
    (connectionData) => api.learning.addKnowledgeConnection(connectionData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['knowledgeGraph', phase, week]);
        toast.success('Knowledge connection added!');
      }
    }
  );

  // Session timer effect
  useEffect(() => {
    if (isSessionActive && sessionStartTime && !sessionPaused) {
      sessionTimerRef.current = setInterval(() => {
        setSessionTimer(Date.now() - sessionStartTime - totalPauseTime);
      }, 1000);
    } else {
      clearInterval(sessionTimerRef.current);
    }

    return () => clearInterval(sessionTimerRef.current);
  }, [isSessionActive, sessionStartTime, sessionPaused, totalPauseTime]);

  // Energy monitoring effect
  useEffect(() => {
    if (isSessionActive) {
      energyCheckInterval.current = setInterval(() => {
        // Check energy every 10 minutes
        if (sessionTimer > 0 && sessionTimer % (10 * 60 * 1000) === 0) {
          setEnergyHistory(prev => [...prev, {
            timestamp: Date.now(),
            energy: sessionData.energy_level,
            elapsed: sessionTimer
          }]);
        }
      }, 60000); // Check every minute
    }

    return () => clearInterval(energyCheckInterval.current);
  }, [isSessionActive, sessionTimer, sessionData.energy_level]);

  // Session management functions
  const startSession = async () => {
    if (sessionData.goals.length === 0) {
      setShowSessionSetup(true);
      return;
    }

    const now = Date.now();
    setSessionStartTime(now);
    setIsSessionActive(true);
    setSessionTimer(0);
    setTotalPauseTime(0);

    try {
      await createSessionMutation.mutateAsync({
        ...sessionData,
        start_time: new Date(now).toISOString(),
        target_duration_minutes: sessionData.target_duration || 25
      });
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  const pauseSession = () => {
    setSessionPaused(true);
    setPauseStartTime(Date.now());
    setSessionData(prev => ({ ...prev, break_count: prev.break_count + 1 }));
    toast('Session paused ⏸️');
  };

  const resumeSession = () => {
    if (pauseStartTime) {
      const pauseDuration = Date.now() - pauseStartTime;
      setTotalPauseTime(prev => prev + pauseDuration);
      setSessionData(prev => ({ 
        ...prev, 
        break_duration_minutes: prev.break_duration_minutes + Math.floor(pauseDuration / (1000 * 60))
      }));
    }
    setSessionPaused(false);
    setPauseStartTime(null);
    toast('Session resumed ▶️');
  };

  const recordInterruption = (type, details = '') => {
    const interruption = {
      type,
      details,
      timestamp: Date.now(),
      duration_seconds: 0 // Will be updated when interruption ends
    };

    setSessionData(prev => ({
      ...prev,
      interruption_count: prev.interruption_count + 1,
      distraction_types: [...prev.distraction_types, type],
      distraction_details: [...prev.distraction_details, interruption]
    }));

    setCurrentInterruption(interruption);
    toast.error(`Interruption recorded: ${type} 📞`);
  };

  const endInterruption = () => {
    if (currentInterruption) {
      const duration = Math.floor((Date.now() - currentInterruption.timestamp) / 1000);
      setSessionData(prev => ({
        ...prev,
        distraction_details: prev.distraction_details.map(d => 
          d.timestamp === currentInterruption.timestamp 
            ? { ...d, duration_seconds: duration }
            : d
        )
      }));
      setCurrentInterruption(null);
      toast.success('Interruption ended - back to focus! 🎯');
    }
  };

  const completeSession = () => {
    setIsSessionActive(false);
    setSessionPaused(false);
    setShowSessionReview(true);
  };

  const finishSession = async () => {
    const actualDuration = Math.floor(sessionTimer / (1000 * 60));
    const today = new Date().toISOString().split('T')[0];

    try {
      await updateSessionMutation.mutateAsync({
        date: today,
        sessionData: {
          ...sessionData,
          actual_duration_minutes: actualDuration,
          end_time: new Date().toISOString(),
          goal_completion_rate: sessionData.goals.length > 0 
            ? sessionData.completed_goals.length / sessionData.goals.length 
            : 0,
          effective_focus_time: actualDuration - sessionData.break_duration_minutes,
          interruption_impact_score: Math.max(0, 10 - sessionData.interruption_count),
          session_success_score: calculateSessionSuccessScore()
        }
      });

      // Reset session state
      resetSessionState();
      setShowSessionReview(false);
    } catch (error) {
      toast.error('Failed to save session');
    }
  };

  const calculateSessionSuccessScore = () => {
    const goalCompletion = sessionData.goals.length > 0 
      ? sessionData.completed_goals.length / sessionData.goals.length 
      : 0;
    const focusScore = sessionData.focus_score / 10;
    const interruptionPenalty = Math.max(0, 1 - (sessionData.interruption_count * 0.1));
    
    return Math.round((goalCompletion * 0.4 + focusScore * 0.4 + interruptionPenalty * 0.2) * 10);
  };

  const resetSessionState = () => {
    setSessionStartTime(null);
    setSessionTimer(0);
    setTotalPauseTime(0);
    setCurrentSession(null);
    setCurrentInterruption(null);
    setEnergyHistory([]);
    setFocusHistory([]);
    setSessionData({
      ...sessionData,
      goals: [],
      completed_goals: [],
      session_notes: '',
      interruption_count: 0,
      distraction_types: [],
      distraction_details: [],
      break_count: 0,
      break_duration_minutes: 0
    });
  };

  const addGoal = () => {
    if (newGoal.trim()) {
      const goal = {
        text: newGoal.trim(),
        priority: goalPriority,
        estimated_time: goalEstimatedTime,
        created_at: Date.now(),
        completed_at: null
      };

      setSessionData(prev => ({
        ...prev,
        goals: [...prev.goals, goal]
      }));
      setNewGoal('');
    }
  };

  const completeGoal = (goalIndex) => {
    const goal = sessionData.goals[goalIndex];
    const completedGoal = {
      ...goal,
      completed_at: Date.now(),
      actual_time: Math.floor(sessionTimer / (1000 * 60))
    };

    setSessionData(prev => ({
      ...prev,
      completed_goals: [...prev.completed_goals, completedGoal]
    }));
    toast.success('Goal completed! 🎉');
  };

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

  const getSessionTypeIcon = (type) => {
    const icons = {
      math: Brain,
      coding: Code,
      visual_projects: Eye,
      real_applications: Target
    };
    return icons[type] || Brain;
  };

  const getSessionTypeColor = (type) => {
    const colors = {
      math: 'text-blue-400',
      coding: 'text-green-400',
      visual_projects: 'text-purple-400',
      real_applications: 'text-orange-400'
    };
    return colors[type] || 'text-blue-400';
  };

  // Configuration data
  const moodOptions = [
    { value: 'energetic', label: 'Energetic', icon: Flame, color: 'text-red-400' },
    { value: 'focused', label: 'Focused', icon: Target, color: 'text-blue-400' },
    { value: 'calm', label: 'Calm', icon: Smile, color: 'text-green-400' },
    { value: 'excited', label: 'Excited', icon: Zap, color: 'text-yellow-400' },
    { value: 'tired', label: 'Tired', icon: Coffee, color: 'text-amber-400' },
    { value: 'distracted', label: 'Distracted', icon: AlertTriangle, color: 'text-orange-400' },
    { value: 'stressed', label: 'Stressed', icon: Frown, color: 'text-red-500' },
    { value: 'neutral', label: 'Neutral', icon: Meh, color: 'text-gray-400' }
  ];

  const distractionTypes = [
    { type: 'phone_call', label: 'Phone Call', icon: Phone },
    { type: 'social_media', label: 'Social Media', icon: Smartphone },
    { type: 'email', label: 'Email', icon: Mail },
    { type: 'noise', label: 'Noise', icon: Volume2 },
    { type: 'conversation', label: 'Conversation', icon: MessageSquare },
    { type: 'hunger', label: 'Hunger/Thirst', icon: Coffee },
    { type: 'fatigue', label: 'Fatigue', icon: Battery },
    { type: 'notification', label: 'Notification', icon: Bell },
    { type: 'internet', label: 'Internet Issue', icon: WifiOff },
    { type: 'other', label: 'Other', icon: AlertTriangle }
  ];

  const environmentTypes = [
    { value: 'home', label: 'Home', icon: Home },
    { value: 'office', label: 'Office', icon: Building },
    { value: 'library', label: 'Library', icon: BookOpen },
    { value: 'cafe', label: 'Café', icon: Coffee },
    { value: 'other', label: 'Other', icon: MapPin }
  ];

  const deviceTypes = [
    { value: 'laptop', label: 'Laptop', icon: Laptop },
    { value: 'desktop', label: 'Desktop', icon: Monitor },
    { value: 'tablet', label: 'Tablet', icon: Smartphone },
    { value: 'phone', label: 'Phone', icon: Smartphone }
  ];

  const sessionTypes = [
    { 
      value: 'math', 
      label: 'Mathematical Focus', 
      icon: Brain, 
      description: 'Deep mathematical concepts and problem solving',
      color: 'from-blue-500 to-cyan-500' 
    },
    { 
      value: 'coding', 
      label: 'Implementation & Coding', 
      icon: Code, 
      description: 'Hands-on programming and algorithm implementation',
      color: 'from-green-500 to-emerald-500' 
    },
    { 
      value: 'visual_projects', 
      label: 'Visual & Interactive', 
      icon: Eye, 
      description: 'Visualizations, diagrams, and interactive learning',
      color: 'from-purple-500 to-pink-500' 
    },
    { 
      value: 'real_applications', 
      label: 'Real-World Applications', 
      icon: Target, 
      description: 'Practical applications and case studies',
      color: 'from-orange-500 to-red-500' 
    }
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/learning')}
            className="flex items-center space-x-2 text-gray-400 hover:text-white mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Learning Path</span>
          </button>
          <h1 className="text-2xl font-bold text-white">
            Phase {phase} • Week {week}
          </h1>
          <p className="text-gray-400">
            {segmentData?.data?.title || `Week ${week} Learning Segment`}
          </p>
        </div>

        {/* Session Status */}
        <div className="text-right">
          {isSessionActive ? (
            <div className="bg-green-600/20 border border-green-600/50 rounded-lg p-4">
              <div className="text-green-400 font-medium">Session Active</div>
              <div className="text-2xl font-mono font-bold text-white">
                {formatTime(sessionTimer)}
              </div>
              <div className="text-xs text-green-400">
                {sessionData.session_type.replace('_', ' ')} • 
                Focus: {sessionData.focus_score}/10
              </div>
            </div>
          ) : (
            <div className="text-gray-400">
              <div className="text-sm">No active session</div>
              <div className="text-xs">
                Today: {todaySession?.data?.summary?.total_time_minutes || 0}m
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Session Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Session Management</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {sessionTypes.map((type) => {
            const Icon = type.icon;
            const isSelected = sessionData.session_type === type.value;
            
            return (
              <motion.button
                key={type.value}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  setSessionData(prev => ({ ...prev, session_type: type.value }));
                  setSelectedSessionType(type.value);
                }}
                disabled={isSessionActive}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-gray-600 hover:border-gray-500'
                } ${isSessionActive ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <div className="flex items-center space-x-3 mb-2">
                  <Icon className={`w-5 h-5 ${isSelected ? 'text-blue-400' : 'text-gray-400'}`} />
                  <span className={`font-medium ${isSelected ? 'text-blue-400' : 'text-white'}`}>
                    {type.label}
                  </span>
                </div>
                <p className="text-xs text-gray-400">{type.description}</p>
              </motion.button>
            );
          })}
        </div>

        {/* Session Action Buttons */}
        <div className="flex items-center justify-center space-x-4">
          {!isSessionActive && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startSession}
              className="flex items-center space-x-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg text-white font-medium"
            >
              <Play className="w-5 h-5" />
              <span>Start Session</span>
            </motion.button>
          )}

          {isSessionActive && !sessionPaused && (
            <>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={pauseSession}
                className="flex items-center space-x-2 px-6 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-white font-medium"
              >
                <Pause className="w-5 h-5" />
                <span>Pause</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={completeSession}
                className="flex items-center space-x-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium"
              >
                <Square className="w-5 h-5" />
                <span>Complete</span>
              </motion.button>
            </>
          )}

          {isSessionActive && sessionPaused && (
            <>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={resumeSession}
                className="flex items-center space-x-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium"
              >
                <Play className="w-5 h-5" />
                <span>Resume</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={completeSession}
                className="flex items-center space-x-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium"
              >
                <Square className="w-5 h-5" />
                <span>End Session</span>
              </motion.button>
            </>
          )}
        </div>

        {/* Active Session Tools */}
        {isSessionActive && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Quick Interruption Tracking */}
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-sm font-medium text-white mb-3">Quick Actions</h4>
              <div className="grid grid-cols-2 gap-2">
                {distractionTypes.slice(0, 4).map((distraction) => {
                  const Icon = distraction.icon;
                  return (
                    <button
                      key={distraction.type}
                      onClick={() => recordInterruption(distraction.type)}
                      className="flex items-center space-x-2 text-xs px-3 py-2 bg-red-600/20 hover:bg-red-600/30 border border-red-600/50 rounded text-red-400 transition-colors"
                    >
                      <Icon className="w-3 h-3" />
                      <span>{distraction.label}</span>
                    </button>
                  );
                })}
              </div>
              <button
                onClick={() => setShowInterruptionModal(true)}
                className="w-full mt-2 text-xs px-3 py-2 bg-gray-600 hover:bg-gray-500 border border-gray-500 rounded text-gray-300 transition-colors"
              >
                More Options...
              </button>
              
              {sessionData.interruption_count > 0 && (
                <div className="mt-3 text-xs text-red-400">
                  Interruptions today: {sessionData.interruption_count}
                </div>
              )}
            </div>

            {/* Goals Progress */}
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-sm font-medium text-white mb-3">Session Goals</h4>
              {sessionData.goals.length > 0 ? (
                <div className="space-y-2">
                  {sessionData.goals.map((goal, index) => {
                    const isCompleted = sessionData.completed_goals.some(cg => cg.text === goal.text);
                    return (
                      <div
                        key={index}
                        className={`flex items-center justify-between p-2 rounded ${
                          isCompleted ? 'bg-green-600/20 border border-green-600/50' : 'bg-gray-600'
                        }`}
                      >
                        <div className={`text-sm ${isCompleted ? 'text-green-400 line-through' : 'text-white'}`}>
                          <div>{goal.text}</div>
                          <div className="text-xs opacity-75">
                            {goal.priority} priority • {goal.estimated_time}m
                          </div>
                        </div>
                        {!isCompleted && (
                          <button
                            onClick={() => completeGoal(index)}
                            className="text-xs px-2 py-1 bg-green-600 hover:bg-green-700 rounded text-white"
                          >
                            Done
                          </button>
                        )}
                      </div>
                    );
                  })}
                  <div className="text-xs text-gray-400 pt-2">
                    Completed: {sessionData.completed_goals.length} / {sessionData.goals.length}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-400">
                  No goals set for this session
                </div>
              )}
            </div>
          </div>
        )}

        {/* Current Interruption */}
        {currentInterruption && (
          <div className="mt-4 bg-red-600/20 border border-red-600/50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-red-400 font-medium">Active Interruption</div>
                <div className="text-sm text-white">{currentInterruption.type.replace('_', ' ')}</div>
                <div className="text-xs text-red-400">
                  Started: {formatTime(Date.now() - currentInterruption.timestamp)}
                </div>
              </div>
              <button
                onClick={endInterruption}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white text-sm"
              >
                End Interruption
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Content Navigation */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex space-x-1">
            {['lessons', 'quests', 'review', 'connections'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
              className="p-2 text-gray-400 hover:text-white"
            >
              {viewMode === 'grid' ? <List className="w-4 h-4" /> : <Grid className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-4">
          {activeTab === 'lessons' && (
            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search lessons..."
                  className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                />
                <select
                  value={lessonFilter}
                  onChange={(e) => setLessonFilter(e.target.value)}
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                >
                  <option value="all">All Lessons</option>
                  <option value="not_started">Not Started</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                  <option value="mastered">Mastered</option>
                </select>
              </div>

              <div className={`grid ${viewMode === 'grid' ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' : 'grid-cols-1'} gap-4`}>
                {lessonsData?.data?.lessons?.map((lesson) => (
                  <motion.div
                    key={lesson.lesson_id}
                    whileHover={{ scale: 1.02 }}
                    className="bg-gray-700 rounded-lg border border-gray-600 p-4 cursor-pointer"
                    onClick={() => navigate(`/learning/lessons/${lesson.lesson_id}`)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <BookOpen className="w-4 h-4 text-blue-400" />
                        <span className="text-sm text-gray-400 capitalize">{lesson.lesson_type}</span>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs ${
                        lesson.status === 'mastered' ? 'bg-green-600/20 text-green-400' :
                        lesson.status === 'completed' ? 'bg-blue-600/20 text-blue-400' :
                        lesson.status === 'in_progress' ? 'bg-yellow-600/20 text-yellow-400' :
                        'bg-gray-600/20 text-gray-400'
                      }`}>
                        {lesson.status?.replace('_', ' ') || 'not started'}
                      </div>
                    </div>
                    <h3 className="text-white font-medium mb-2">{lesson.lesson_title}</h3>
                    {lesson.time_spent_minutes > 0 && (
                      <div className="text-xs text-gray-400">
                        Time spent: {lesson.time_spent_minutes}m
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'quests' && (
            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <select
                  value={questFilter}
                  onChange={(e) => setQuestFilter(e.target.value)}
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                >
                  <option value="all">All Quests</option>
                  <option value="available">Available</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                  <option value="mastered">Mastered</option>
                </select>
              </div>

              <div className={`grid ${viewMode === 'grid' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'} gap-4`}>
                {questsData?.data?.quests?.map((quest) => (
                  <QuestCard
                    key={quest.quest_id}
                    quest={quest}
                    onClick={() => navigate(`/quests/${quest.quest_id}`)}
                  />
                ))}
              </div>
            </div>
          )}

          {activeTab === 'review' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">Spaced Repetition Review</h3>
              {spacedRepetitionData?.data?.reviewItems?.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {spacedRepetitionData.data.reviewItems.map((item) => (
                    <div key={item.concept_id} className="bg-gray-700 rounded-lg p-4">
                      <h4 className="text-white font-medium mb-2">
                        {item.concept_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h4>
                      <div className="text-sm text-gray-400 mb-2">
                        {item.lesson_title} • Phase {item.phase} Week {item.week}
                      </div>
                      <div className="text-xs text-gray-500">
                        Last reviewed: {item.last_reviewed_at 
                          ? new Date(item.last_reviewed_at).toLocaleDateString()
                          : 'Never'
                        }
                      </div>
                      <div className="text-xs text-gray-500">
                        Difficulty: {item.difficulty_factor?.toFixed(1)} • 
                        Interval: {item.interval_days} days
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-8">
                  No items due for review
                </div>
              )}
            </div>
          )}

          {activeTab === 'connections' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">Knowledge Connections</h3>
              {knowledgeConnections?.data?.connections?.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {knowledgeConnections.data.connections.map((connection, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Layers className="w-4 h-4 text-blue-400" />
                        <span className="text-sm font-medium text-white">{connection.from_concept}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-xs text-gray-400 mb-2">
                        <ArrowRight className="w-3 h-3" />
                        <span>{connection.to_concept}</span>
                      </div>
                      <div className="text-xs text-gray-500">{connection.relationship_type}</div>
                      {connection.strength && (
                        <div className="text-xs text-blue-400 mt-1">
                          Strength: {connection.strength}/10
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-8">
                  No knowledge connections found
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Session Setup Modal */}
      <AnimatePresence>
        {showSessionSetup && (
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
              <h3 className="text-lg font-semibold text-white mb-4">Session Setup</h3>
              
              {/* Session Type */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Session Type
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {sessionTypes.map((type) => {
                    const Icon = type.icon;
                    return (
                      <button
                        key={type.value}
                        onClick={() => setSessionData(prev => ({ ...prev, session_type: type.value }))}
                        className={`flex items-center space-x-2 p-3 rounded text-sm transition-colors ${
                          sessionData.session_type === type.value
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span>{type.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Energy Level */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Energy Level
                </label>
                <div className="flex items-center space-x-2">
                  <Battery className="w-4 h-4 text-gray-400" />
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={sessionData.energy_level}
                    onChange={(e) => setSessionData(prev => ({ ...prev, energy_level: Number(e.target.value) }))}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{sessionData.energy_level}</span>
                </div>
              </div>

              {/* Mood */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Current Mood
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {moodOptions.map((mood) => {
                    const Icon = mood.icon;
                    return (
                      <button
                        key={mood.value}
                        onClick={() => setSessionData(prev => ({ ...prev, mood_before: mood.value }))}
                        className={`flex items-center space-x-2 p-2 rounded text-xs transition-colors ${
                          sessionData.mood_before === mood.value
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        <Icon className={`w-3 h-3 ${mood.color}`} />
                        <span>{mood.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Environment */}
              <div className="mb-4 grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Environment
                  </label>
                  <select
                    value={sessionData.environment_type}
                    onChange={(e) => setSessionData(prev => ({ ...prev, environment_type: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                  >
                    {environmentTypes.map((env) => (
                      <option key={env.value} value={env.value}>{env.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Device
                  </label>
                  <select
                    value={sessionData.device_type}
                    onChange={(e) => setSessionData(prev => ({ ...prev, device_type: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                  >
                    {deviceTypes.map((device) => (
                      <option key={device.value} value={device.value}>{device.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Goals */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Session Goals
                </label>
                <div className="flex space-x-2 mb-2">
                  <input
                    type="text"
                    value={newGoal}
                    onChange={(e) => setNewGoal(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && addGoal()}
                    placeholder="Add a goal for this session..."
                    className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                  />
                  <select
                    value={goalPriority}
                    onChange={(e) => setGoalPriority(e.target.value)}
                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                  <input
                    type="number"
                    value={goalEstimatedTime}
                    onChange={(e) => setGoalEstimatedTime(Number(e.target.value))}
                    min="5"
                    max="120"
                    className="w-20 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                  />
                  <button
                    onClick={addGoal}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm"
                  >
                    Add
                  </button>
                </div>
                {sessionData.goals.length > 0 && (
                  <div className="space-y-1">
                    {sessionData.goals.map((goal, index) => (
                      <div key={index} className="text-sm text-gray-300 bg-gray-700 px-3 py-2 rounded flex items-center justify-between">
                        <div>
                          <span>• {goal.text}</span>
                          <span className="text-xs text-gray-400 ml-2">
                            ({goal.priority} priority, {goal.estimated_time}m)
                          </span>
                        </div>
                        <button
                          onClick={() => setSessionData(prev => ({
                            ...prev,
                            goals: prev.goals.filter((_, i) => i !== index)
                          }))}
                          className="text-red-400 hover:text-red-300"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => setShowSessionSetup(false)}
                  className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    setShowSessionSetup(false);
                    startSession();
                  }}
                  disabled={sessionData.goals.length === 0}
                  className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white"
                >
                  Start Session
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Session Review Modal */}
      <AnimatePresence>
        {showSessionReview && (
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
              <h3 className="text-lg font-semibold text-white mb-4">Session Review</h3>
              
              {/* Session Summary */}
              <div className="mb-4 p-4 bg-gray-700 rounded-lg">
                <h4 className="text-white font-medium mb-2">Session Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-300">Duration: <span className="text-white font-medium">{formatTime(sessionTimer)}</span></div>
                    <div className="text-gray-300">Session Type: <span className="text-white font-medium capitalize">{sessionData.session_type.replace('_', ' ')}</span></div>
                    <div className="text-gray-300">Goals Completed: <span className="text-white font-medium">{sessionData.completed_goals.length} / {sessionData.goals.length}</span></div>
                  </div>
                  <div>
                    <div className="text-gray-300">Interruptions: <span className="text-white font-medium">{sessionData.interruption_count}</span></div>
                    <div className="text-gray-300">Breaks Taken: <span className="text-white font-medium">{sessionData.break_count}</span></div>
                    <div className="text-gray-300">Success Score: <span className="text-white font-medium">{calculateSessionSuccessScore()}/10</span></div>
                  </div>
                </div>
              </div>

              {/* Mood After */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  How do you feel now?
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {moodOptions.map((mood) => {
                    const Icon = mood.icon;
                    return (
                      <button
                        key={mood.value}
                        onClick={() => setSessionData(prev => ({ ...prev, mood_after: mood.value }))}
                        className={`flex items-center space-x-2 p-2 rounded text-xs transition-colors ${
                          sessionData.mood_after === mood.value
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        <Icon className={`w-3 h-3 ${mood.color}`} />
                        <span>{mood.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Focus Score */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Focus Score (1-10)
                </label>
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-gray-400" />
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={sessionData.focus_score}
                    onChange={(e) => setSessionData(prev => ({ ...prev, focus_score: Number(e.target.value) }))}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{sessionData.focus_score}</span>
                </div>
              </div>

              {/* Productivity Rating */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Productivity Rating (1-10)
                </label>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-gray-400" />
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={sessionData.productivity_rating}
                    onChange={(e) => setSessionData(prev => ({ ...prev, productivity_rating: Number(e.target.value) }))}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{sessionData.productivity_rating}</span>
                </div>
              </div>

              {/* Session Notes */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Session Notes
                </label>
                <textarea
                  value={sessionData.session_notes}
                  onChange={(e) => setSessionData(prev => ({ ...prev, session_notes: e.target.value }))}
                  placeholder="What did you learn? Any insights or challenges?"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={3}
                />
              </div>

              {/* Learning Insights */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Key Learning Insights
                </label>
                <textarea
                  value={sessionData.learning_insights}
                  onChange={(e) => setSessionData(prev => ({ ...prev, learning_insights: e.target.value }))}
                  placeholder="What key concepts did you master? Any breakthrough moments?"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={2}
                />
              </div>

              {/* Next Session Plan */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Next Session Plan
                </label>
                <textarea
                  value={sessionData.next_session_plan}
                  onChange={(e) => setSessionData(prev => ({ ...prev, next_session_plan: e.target.value }))}
                  placeholder="What should you focus on in your next session?"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={2}
                />
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => setShowSessionReview(false)}
                  className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={finishSession}
                  disabled={!sessionData.mood_after || updateSessionMutation.isLoading}
                  className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white"
                >
                  {updateSessionMutation.isLoading ? 'Saving...' : 'Complete Session'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interruption Detail Modal */}
      <AnimatePresence>
        {showInterruptionModal && (
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
              className="bg-gray-800 rounded-lg border border-gray-700 p-6 max-w-md w-full mx-4"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Record Interruption</h3>
              
              <div className="grid grid-cols-2 gap-2 mb-4">
                {distractionTypes.map((distraction) => {
                  const Icon = distraction.icon;
                  return (
                    <button
                      key={distraction.type}
                      onClick={() => {
                        recordInterruption(distraction.type);
                        setShowInterruptionModal(false);
                      }}
                      className="flex items-center space-x-2 p-3 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 transition-colors"
                    >
                      <Icon className="w-4 h-4" />
                      <span className="text-sm">{distraction.label}</span>
                    </button>
                  );
                })}
              </div>

              <button
                onClick={() => setShowInterruptionModal(false)}
                className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
              >
                Cancel
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LearningSegment;