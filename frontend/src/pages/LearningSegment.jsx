/**
 * Enhanced Neural Odyssey Learning Segment Component
 * 
 * Now leverages the rich session management capabilities from the backend:
 * - Energy level tracking (1-10)
 * - Mood before/after session tracking
 * - Focus score monitoring (1-10) 
 * - Interruption and distraction tracking
 * - Goal setting and completion tracking
 * - Session notes and reflection
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
  Minus
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import QuestCard from '../components/QuestCard';
import CodePlayground from '../components/CodePlayground';
import VaultRevealModal from '../components/VaultRevealModal';

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
  
  // Enhanced Session Data
  const [sessionData, setSessionData] = useState({
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

  // Session Setup Modal
  const [showSessionSetup, setShowSessionSetup] = useState(false);
  const [showSessionReview, setShowSessionReview] = useState(false);
  const [selectedSessionType, setSelectedSessionType] = useState(null);

  // Other State
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'lessons');
  const [selectedLesson, setSelectedLesson] = useState(null);
  const [selectedQuest, setSelectedQuest] = useState(null);
  const [showCodePlayground, setShowCodePlayground] = useState(false);
  const [lessonFilter, setLessonFilter] = useState('all');
  const [questFilter, setQuestFilter] = useState('all');
  const [viewMode, setViewMode] = useState('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [newVaultUnlock, setNewVaultUnlock] = useState(null);

  // Parse URL params
  const currentPhase = parseInt(phase) || 1;
  const currentWeek = parseInt(week) || 1;
  const sessionType = searchParams.get('session');
  const questId = searchParams.get('quest');

  // Refs
  const interruptionTimeoutRef = useRef(null);

  // Fetch data
  const { data: progressData, isLoading: progressLoading } = useQuery(
    ['learningProgress', currentPhase, currentWeek],
    () => api.get(`/learning/progress?phase=${currentPhase}&week=${currentWeek}`),
    { refetchInterval: 30000 }
  );

  const { data: questsData, isLoading: questsLoading } = useQuery(
    ['quests', currentPhase, currentWeek],
    () => api.get(`/learning/quests?phase=${currentPhase}&week=${currentWeek}`),
    { refetchInterval: 30000 }
  );

  const { data: todayData } = useQuery(
    'todaySessions',
    () => api.get('/learning/sessions/today'),
    { refetchInterval: 60000 }
  );

  // Session mutations
  const startSessionMutation = useMutation(
    (sessionData) => api.post('/learning/sessions', sessionData),
    {
      onSuccess: (data) => {
        setCurrentSession(data.data);
        setIsSessionActive(true);
        setSessionTimer(0);
        setSessionStartTime(Date.now());
        toast.success('Learning session started!');
        setShowSessionSetup(false);
      }
    }
  );

  const endSessionMutation = useMutation(
    ({ sessionId, sessionData }) => api.put(`/learning/sessions/${sessionId}/end`, sessionData),
    {
      onSuccess: () => {
        setCurrentSession(null);
        setIsSessionActive(false);
        setSessionTimer(0);
        setSessionStartTime(null);
        queryClient.invalidateQueries('todaySessions');
        toast.success('Session completed!');
        setShowSessionReview(false);
        // Reset session data
        setSessionData({
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
      }
    }
  );

  // Timer effect
  useEffect(() => {
    let interval;
    if (isSessionActive) {
      interval = setInterval(() => {
        setSessionTimer(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isSessionActive]);

  // Session type options with enhanced descriptions
  const sessionTypes = [
    { 
      id: 'math', 
      label: 'Mathematics', 
      icon: Brain, 
      color: 'blue',
      description: 'Deep mathematical concepts and theory',
      defaultGoals: ['Understand core mathematical concepts', 'Complete theory exercises', 'Build mathematical intuition']
    },
    { 
      id: 'coding', 
      label: 'Coding Practice', 
      icon: Code, 
      color: 'green',
      description: 'Hands-on programming and implementation',
      defaultGoals: ['Write clean, working code', 'Debug and test solutions', 'Practice programming patterns']
    },
    { 
      id: 'visual_projects', 
      label: 'Visual Projects', 
      icon: Eye, 
      color: 'purple',
      description: 'Data visualization and visual learning',
      defaultGoals: ['Create meaningful visualizations', 'Understand visual patterns', 'Build interactive demos']
    },
    { 
      id: 'real_applications', 
      label: 'Real Applications', 
      icon: Target, 
      color: 'orange',
      description: 'Practical real-world applications',
      defaultGoals: ['Apply concepts to real problems', 'Build practical solutions', 'Connect theory to practice']
    }
  ];

  // Mood options
  const moodOptions = [
    { id: 'excited', label: 'Excited', icon: Smile, color: 'text-yellow-400' },
    { id: 'focused', label: 'Focused', icon: Target, color: 'text-blue-400' },
    { id: 'calm', label: 'Calm', icon: Meh, color: 'text-green-400' },
    { id: 'tired', label: 'Tired', icon: Coffee, color: 'text-orange-400' },
    { id: 'distracted', label: 'Distracted', icon: AlertTriangle, color: 'text-red-400' }
  ];

  // Distraction types
  const distractionTypes = [
    { id: 'phone', label: 'Phone', icon: Phone },
    { id: 'email', label: 'Email', icon: Mail },
    { id: 'noise', label: 'Noise', icon: Volume2 },
    { id: 'interruption', label: 'Interruption', icon: Users },
    { id: 'fatigue', label: 'Fatigue', icon: Battery },
    { id: 'other', label: 'Other', icon: AlertTriangle }
  ];

  // Handle session start
  const handleStartSession = (type) => {
    if (currentSession) {
      toast.error('Please end current session first');
      return;
    }
    setSelectedSessionType(type);
    const typeInfo = sessionTypes.find(t => t.id === type);
    setSessionData(prev => ({
      ...prev,
      goals: typeInfo.defaultGoals
    }));
    setShowSessionSetup(true);
  };

  // Handle session setup completion
  const handleSessionSetupComplete = () => {
    startSessionMutation.mutate({
      session_type: selectedSessionType,
      planned_duration_minutes: 25,
      energy_level: sessionData.energy_level,
      mood_before: sessionData.mood_before,
      goals: JSON.stringify(sessionData.goals)
    });
  };

  // Handle session end
  const handleEndSession = () => {
    if (!currentSession) return;
    setShowSessionReview(true);
  };

  // Handle session review completion
  const handleSessionReviewComplete = () => {
    const duration = Math.floor(sessionTimer / 60);
    endSessionMutation.mutate({
      sessionId: currentSession.id,
      sessionData: {
        end_time: new Date().toISOString(),
        actual_duration_minutes: duration,
        focus_score: sessionData.focus_score,
        mood_after: sessionData.mood_after,
        interruption_count: sessionData.interruption_count,
        distraction_types: JSON.stringify(sessionData.distraction_types),
        completed_goals: JSON.stringify(sessionData.completed_goals),
        session_notes: sessionData.session_notes
      }
    });
  };

  // Track interruption
  const handleInterruption = (type) => {
    setSessionData(prev => ({
      ...prev,
      interruption_count: prev.interruption_count + 1,
      distraction_types: [...prev.distraction_types, type].filter((v, i, a) => a.indexOf(v) === i)
    }));
    
    // Brief pause suggestion
    clearTimeout(interruptionTimeoutRef.current);
    interruptionTimeoutRef.current = setTimeout(() => {
      toast.info('Consider taking a short break to refocus', { duration: 3000 });
    }, 2000);
  };

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Render session setup modal
  const renderSessionSetup = () => {
    if (!showSessionSetup) return null;

    const typeInfo = sessionTypes.find(t => t.id === selectedSessionType);
    const IconComponent = typeInfo?.icon || Brain;

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
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-md w-full max-h-[80vh] overflow-y-auto"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-2 rounded-lg bg-${typeInfo?.color}-500 bg-opacity-20`}>
                <IconComponent className={`w-6 h-6 text-${typeInfo?.color}-400`} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">{typeInfo?.label}</h3>
                <p className="text-sm text-gray-400">{typeInfo?.description}</p>
              </div>
            </div>

            {/* Energy Level */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-white mb-2">
                Energy Level: {sessionData.energy_level}/10
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={sessionData.energy_level}
                onChange={(e) => setSessionData(prev => ({
                  ...prev,
                  energy_level: parseInt(e.target.value)
                }))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Low</span>
                <span>High</span>
              </div>
            </div>

            {/* Mood Before */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-white mb-2">
                How are you feeling?
              </label>
              <div className="grid grid-cols-3 gap-2">
                {moodOptions.map(mood => {
                  const MoodIcon = mood.icon;
                  return (
                    <button
                      key={mood.id}
                      onClick={() => setSessionData(prev => ({
                        ...prev,
                        mood_before: mood.id
                      }))}
                      className={`p-2 rounded-lg border transition-colors ${
                        sessionData.mood_before === mood.id
                          ? 'border-blue-500 bg-blue-500 bg-opacity-20'
                          : 'border-gray-600 hover:border-gray-500'
                      }`}
                    >
                      <MoodIcon className={`w-4 h-4 mx-auto mb-1 ${mood.color}`} />
                      <div className="text-xs text-white">{mood.label}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Goals */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-white mb-2">
                Session Goals
              </label>
              <div className="space-y-2">
                {sessionData.goals.map((goal, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <input
                      type="text"
                      value={goal}
                      onChange={(e) => {
                        const newGoals = [...sessionData.goals];
                        newGoals[index] = e.target.value;
                        setSessionData(prev => ({ ...prev, goals: newGoals }));
                      }}
                      className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                      placeholder="Enter a goal for this session"
                    />
                    <button
                      onClick={() => {
                        const newGoals = sessionData.goals.filter((_, i) => i !== index);
                        setSessionData(prev => ({ ...prev, goals: newGoals }));
                      }}
                      className="p-2 text-red-400 hover:bg-red-500 hover:bg-opacity-20 rounded"
                    >
                      <Minus className="w-4 h-4" />
                    </button>
                  </div>
                ))}
                <button
                  onClick={() => {
                    setSessionData(prev => ({
                      ...prev,
                      goals: [...prev.goals, '']
                    }));
                  }}
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm"
                >
                  <Plus className="w-4 h-4" />
                  Add Goal
                </button>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={() => setShowSessionSetup(false)}
                className="flex-1 px-4 py-2 border border-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSessionSetupComplete}
                disabled={startSessionMutation.isLoading}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
              >
                {startSessionMutation.isLoading ? 'Starting...' : 'Start Session'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    );
  };

  // Render session review modal
  const renderSessionReview = () => {
    if (!showSessionReview) return null;

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
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-md w-full max-h-[80vh] overflow-y-auto"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Session Review</h3>

            {/* Session Summary */}
            <div className="mb-4 p-3 bg-gray-700 rounded-lg">
              <div className="text-sm text-gray-400">Session Duration</div>
              <div className="text-xl font-mono text-white">{formatTime(sessionTimer)}</div>
            </div>

            {/* Focus Score */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-white mb-2">
                Focus Score: {sessionData.focus_score}/10
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={sessionData.focus_score}
                onChange={(e) => setSessionData(prev => ({
                  ...prev,
                  focus_score: parseInt(e.target.value)
                }))}
                className="w-full"
              />
            </div>

            {/* Mood After */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-white mb-2">
                How do you feel now?
              </label>
              <div className="grid grid-cols-3 gap-2">
                {moodOptions.map(mood => {
                  const MoodIcon = mood.icon;
                  return (
                    <button
                      key={mood.id}
                      onClick={() => setSessionData(prev => ({
                        ...prev,
                        mood_after: mood.id
                      }))}
                      className={`p-2 rounded-lg border transition-colors ${
                        sessionData.mood_after === mood.id
                          ? 'border-blue-500 bg-blue-500 bg-opacity-20'
                          : 'border-gray-600 hover:border-gray-500'
                      }`}
                    >
                      <MoodIcon className={`w-4 h-4 mx-auto mb-1 ${mood.color}`} />
                      <div className="text-xs text-white">{mood.label}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Completed Goals */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-white mb-2">
                Completed Goals
              </label>
              <div className="space-y-2">
                {sessionData.goals.map((goal, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={sessionData.completed_goals.includes(goal)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSessionData(prev => ({
                            ...prev,
                            completed_goals: [...prev.completed_goals, goal]
                          }));
                        } else {
                          setSessionData(prev => ({
                            ...prev,
                            completed_goals: prev.completed_goals.filter(g => g !== goal)
                          }));
                        }
                      }}
                      className="rounded border-gray-600"
                    />
                    <span className="text-sm text-white">{goal}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Session Notes */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-white mb-2">
                Session Notes
              </label>
              <textarea
                value={sessionData.session_notes}
                onChange={(e) => setSessionData(prev => ({
                  ...prev,
                  session_notes: e.target.value
                }))}
                className="w-full h-20 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm resize-none"
                placeholder="What did you learn? Any insights or challenges?"
              />
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={() => setShowSessionReview(false)}
                className="flex-1 px-4 py-2 border border-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSessionReviewComplete}
                disabled={endSessionMutation.isLoading}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
              >
                {endSessionMutation.isLoading ? 'Saving...' : 'Complete Session'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    );
  };

  if (progressLoading) {
    return (
      <div className="learning-segment h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-blue-400">
          <BookOpen className="w-6 h-6 animate-pulse" />
          <span>Loading learning content...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="learning-segment h-full overflow-auto"
    >
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex items-center justify-between mb-6"
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            
            <div>
              <h1 className="text-3xl font-bold text-white">
                Phase {currentPhase} - Week {currentWeek}
              </h1>
              <p className="text-gray-400">
                {currentPhase === 1 ? 'Mathematical Foundations and Historical Context' :
                 currentPhase === 2 ? 'Core Machine Learning with Deep Understanding' :
                 currentPhase === 3 ? 'Advanced Topics and Modern AI' :
                 'Mastery and Innovation'}
              </p>
            </div>
          </div>

          {/* Enhanced Session Timer */}
          {currentSession && (
            <div className="flex items-center gap-4">
              <div className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isSessionActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                  <span className="text-white font-mono text-lg">{formatTime(sessionTimer)}</span>
                </div>
                <div className="text-xs text-gray-400 text-center capitalize">
                  {currentSession.session_type?.replace('_', ' ')}
                </div>
              </div>

              {/* Session Controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsSessionActive(!isSessionActive)}
                  className="p-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors text-white"
                >
                  {isSessionActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>

                {/* Quick interruption buttons */}
                <div className="flex gap-1">
                  {distractionTypes.slice(0, 3).map(distraction => {
                    const Icon = distraction.icon;
                    return (
                      <button
                        key={distraction.id}
                        onClick={() => handleInterruption(distraction.id)}
                        className="p-1 text-gray-400 hover:text-orange-400 hover:bg-orange-500 hover:bg-opacity-20 rounded transition-colors"
                        title={`Mark ${distraction.label} interruption`}
                      >
                        <Icon className="w-3 h-3" />
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={handleEndSession}
                  className="p-2 bg-red-500 hover:bg-red-600 rounded-lg transition-colors text-white"
                >
                  <Square className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </motion.div>

        {/* Session Type Selection */}
        {!currentSession && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="mb-8"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Start a Learning Session</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {sessionTypes.map(type => {
                const IconComponent = type.icon;
                return (
                  <button
                    key={type.id}
                    onClick={() => handleStartSession(type.id)}
                    className={`p-4 rounded-lg border-2 border-gray-700 hover:border-${type.color}-500 transition-colors group`}
                  >
                    <div className={`flex items-center gap-3 mb-2`}>
                      <div className={`p-2 rounded-lg bg-${type.color}-500 bg-opacity-20 group-hover:bg-opacity-30 transition-colors`}>
                        <IconComponent className={`w-5 h-5 text-${type.color}-400`} />
                      </div>
                      <span className="font-medium text-white">{type.label}</span>
                    </div>
                    <p className="text-sm text-gray-400 text-left">{type.description}</p>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* Session interruption tracking display */}
        {currentSession && sessionData.interruption_count > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 bg-orange-500 bg-opacity-10 border border-orange-500 border-opacity-30 rounded-lg"
          >
            <div className="flex items-center gap-2 text-orange-400">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">
                {sessionData.interruption_count} interruption{sessionData.interruption_count !== 1 ? 's' : ''} logged
              </span>
            </div>
          </motion.div>
        )}

        {/* Rest of the component remains the same... */}
        {/* Tab Navigation, Content, etc. */}
        
      </div>

      {/* Session Setup Modal */}
      {renderSessionSetup()}

      {/* Session Review Modal */}
      {renderSessionReview()}

      {/* Vault Reveal Modal */}
      <VaultRevealModal
        vaultItem={newVaultUnlock}
        isOpen={!!newVaultUnlock}
        onClose={() => setNewVaultUnlock(null)}
      />
    </motion.div>
  );
};

export default LearningSegment;