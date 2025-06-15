import { useState, useEffect } from 'react';
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
  Users
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

  // State management
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'lessons');
  const [selectedLesson, setSelectedLesson] = useState(null);
  const [selectedQuest, setSelectedQuest] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);
  const [sessionTimer, setSessionTimer] = useState(0);
  const [isSessionActive, setIsSessionActive] = useState(false);
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

  const { data: reviewData } = useQuery(
    'reviewItems',
    () => api.get('/learning/review'),
    { refetchInterval: 60000 }
  );

  // Session mutations
  const startSessionMutation = useMutation(
    (sessionData) => api.post('/learning/sessions/start', sessionData),
    {
      onSuccess: (data) => {
        setCurrentSession(data.data);
        setIsSessionActive(true);
        setSessionTimer(0);
        toast.success('Learning session started!');
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
        queryClient.invalidateQueries('todaySessions');
        toast.success('Session completed!');
      }
    }
  );

  // Progress update mutation
  const updateProgressMutation = useMutation(
    ({ lessonId, progressData }) => api.put(`/learning/progress/${lessonId}`, progressData),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('learningProgress');
        
        // Check for vault unlocks
        if (data.data.vault_unlocks && data.data.vault_unlocks.length > 0) {
          setNewVaultUnlock(data.data.vault_unlocks[0]);
        }
        
        toast.success('Progress updated!');
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

  // Handle session start
  const handleStartSession = (type) => {
    if (currentSession) {
      toast.error('Please end current session first');
      return;
    }

    startSessionMutation.mutate({
      session_type: type,
      energy_level: 8, // Default energy level
      goals: [`Complete ${type} practice session`]
    });
  };

  // Handle session end
  const handleEndSession = () => {
    if (!currentSession) return;

    const duration = Math.floor(sessionTimer / 60);
    endSessionMutation.mutate({
      sessionId: currentSession.session_id,
      sessionData: {
        focus_score: 8, // Could be user input
        session_notes: `Completed ${duration} minute session`,
        completed_goals: [`Studied for ${duration} minutes`]
      }
    });
  };

  // Handle lesson completion
  const handleLessonComplete = (lesson) => {
    updateProgressMutation.mutate({
      lessonId: lesson.lesson_id,
      progressData: {
        status: 'completed',
        completion_percentage: 100,
        time_spent_minutes: 30 // Default time
      }
    });
  };

  // Handle quest start
  const handleQuestStart = (quest) => {
    setSelectedQuest(quest);
    setShowCodePlayground(true);
    setActiveTab('coding');
  };

  // Filter lessons
  const filteredLessons = progressData?.data?.progress?.filter(lesson => {
    if (lessonFilter !== 'all' && lesson.lesson_type !== lessonFilter) return false;
    if (searchQuery && !lesson.lesson_title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }) || [];

  // Filter quests
  const filteredQuests = questsData?.data?.filter(quest => {
    if (questFilter !== 'all' && quest.type !== questFilter) return false;
    if (searchQuery && !quest.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  }) || [];

  // Get lesson type info
  const getLessonTypeInfo = (type) => {
    const types = {
      theory: { icon: BookOpen, label: 'Theory', color: 'text-blue-400', bg: 'bg-blue-400/10' },
      math: { icon: Brain, label: 'Mathematics', color: 'text-green-400', bg: 'bg-green-400/10' },
      visual: { icon: Eye, label: 'Visual', color: 'text-purple-400', bg: 'bg-purple-400/10' },
      coding: { icon: Code, label: 'Coding', color: 'text-orange-400', bg: 'bg-orange-400/10' }
    };
    return types[type] || types.theory;
  };

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Get session type options
  const sessionTypes = [
    { id: 'math', label: 'Mathematics', icon: Brain, color: 'blue' },
    { id: 'coding', label: 'Coding Practice', icon: Code, color: 'green' },
    { id: 'visual_projects', label: 'Visual Projects', icon: Eye, color: 'purple' },
    { id: 'real_applications', label: 'Real Applications', icon: Target, color: 'orange' }
  ];

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

          {/* Session Timer */}
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
              
              <button
                onClick={() => setIsSessionActive(!isSessionActive)}
                className="p-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors text-white"
              >
                {isSessionActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              
              <button
                onClick={handleEndSession}
                className="p-2 bg-red-500 hover:bg-red-600 rounded-lg transition-colors text-white"
              >
                <Square className="w-4 h-4" />
              </button>
            </div>
          )}
        </motion.div>

        {/* Navigation Tabs */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="flex items-center justify-between mb-6"
        >
          <div className="flex items-center gap-2 bg-gray-800 p-1 rounded-lg">
            {[
              { id: 'lessons', label: 'Lessons', icon: BookOpen },
              { id: 'quests', label: 'Quests', icon: Target },
              { id: 'coding', label: 'Code', icon: Code },
              { id: 'review', label: 'Review', icon: Lightbulb }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setSearchParams(prev => ({ ...prev, tab: tab.id }));
                }}
                className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                  activeTab === tab.id 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Quick Session Start */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">Quick Start:</span>
            {sessionTypes.map(type => (
              <button
                key={type.id}
                onClick={() => handleStartSession(type.id)}
                disabled={!!currentSession || startSessionMutation.isLoading}
                className={`p-2 hover:bg-${type.color}-500/20 rounded-lg transition-colors text-${type.color}-400 disabled:opacity-50 disabled:cursor-not-allowed`}
                title={type.label}
              >
                <type.icon className="w-4 h-4" />
              </button>
            ))}
          </div>
        </motion.div>

        {/* Content Area */}
        <AnimatePresence mode="wait">
          {/* Lessons Tab */}
          {activeTab === 'lessons' && (
            <motion.div
              key="lessons"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Filters */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search lessons..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-400 focus:outline-none"
                    />
                  </div>
                  
                  <select
                    value={lessonFilter}
                    onChange={(e) => setLessonFilter(e.target.value)}
                    className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                  >
                    <option value="all">All Types</option>
                    <option value="theory">Theory</option>
                    <option value="math">Mathematics</option>
                    <option value="visual">Visual</option>
                    <option value="coding">Coding</option>
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-2 rounded-lg transition-colors ${
                      viewMode === 'grid' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`p-2 rounded-lg transition-colors ${
                      viewMode === 'list' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Lessons Grid */}
              <div className={`grid gap-4 ${
                viewMode === 'grid' 
                  ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' 
                  : 'grid-cols-1'
              }`}>
                {filteredLessons.map((lesson, index) => {
                  const typeInfo = getLessonTypeInfo(lesson.lesson_type);
                  const IconComponent = typeInfo.icon;
                  
                  return (
                    <motion.div
                      key={lesson.lesson_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4 hover:border-gray-600 transition-all duration-300"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${typeInfo.bg}`}>
                            <IconComponent className={`w-5 h-5 ${typeInfo.color}`} />
                          </div>
                          <div>
                            <h3 className="font-semibold text-white">{lesson.lesson_title}</h3>
                            <span className={`text-sm ${typeInfo.color}`}>{typeInfo.label}</span>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-1">
                          {lesson.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-400" />}
                          {lesson.status === 'mastered' && <Star className="w-4 h-4 text-yellow-400 fill-current" />}
                        </div>
                      </div>

                      {/* Progress Bar */}
                      <div className="mb-3">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-400">Progress</span>
                          <span className="text-white">{lesson.completion_percentage || 0}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full transition-all duration-300"
                            style={{ 
                              width: `${lesson.completion_percentage || 0}%`,
                              background: `linear-gradient(90deg, ${typeInfo.color.replace('text-', '')}, ${typeInfo.color.replace('text-', '')}dd)`
                            }}
                          />
                        </div>
                      </div>

                      {/* Action Button */}
                      <button
                        onClick={() => handleLessonComplete(lesson)}
                        disabled={updateProgressMutation.isLoading}
                        className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                      >
                        {lesson.status === 'completed' ? (
                          <>
                            <Trophy className="w-4 h-4" />
                            Mark as Mastered
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            Start Lesson
                          </>
                        )}
                      </button>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* Quests Tab */}
          {activeTab === 'quests' && (
            <motion.div
              key="quests"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Quest Filters */}
              <div className="flex items-center gap-4">
                <select
                  value={questFilter}
                  onChange={(e) => setQuestFilter(e.target.value)}
                  className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                >
                  <option value="all">All Quest Types</option>
                  <option value="coding_exercise">Coding Exercises</option>
                  <option value="implementation_project">Implementation Projects</option>
                  <option value="theory_quiz">Theory Quizzes</option>
                  <option value="practical_application">Practical Applications</option>
                </select>
              </div>

              {/* Quests Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {filteredQuests.map((quest, index) => (
                  <QuestCard
                    key={quest.id}
                    quest={quest}
                    onStart={handleQuestStart}
                  />
                ))}
              </div>
            </motion.div>
          )}

          {/* Coding Tab */}
          {activeTab === 'coding' && (
            <motion.div
              key="coding"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="h-[calc(100vh-200px)]"
            >
              <CodePlayground
                quest={selectedQuest}
                onCodeSubmit={(code, output, testResults) => {
                  // Handle quest submission
                  toast.success('Quest completed successfully!');
                }}
              />
            </motion.div>
          )}

          {/* Review Tab */}
          {activeTab === 'review' && (
            <motion.div
              key="review"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-2">Spaced Repetition Review</h2>
                <p className="text-gray-400">
                  Reinforce your learning with scientifically-optimized review sessions
                </p>
              </div>

              {reviewData?.data?.review_items && reviewData.data.review_items.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {reviewData.data.review_items.map((item, index) => (
                    <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                      <h3 className="font-semibold text-white mb-2">{item.concept_title}</h3>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1">
                          {Array.from({ length: 5 }, (_, i) => (
                            <Star
                              key={i}
                              className={`w-3 h-3 ${
                                i < Math.floor(item.difficulty_factor) 
                                  ? 'fill-yellow-400 text-yellow-400' 
                                  : 'text-gray-600'
                              }`}
                            />
                          ))}
                        </div>
                        <span className="text-sm text-gray-400">
                          Review #{item.repetition_count + 1}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Lightbulb className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-400 mb-2">No reviews due</h3>
                  <p className="text-gray-500">Great job! Check back tomorrow for new review items.</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Navigation */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex items-center justify-between mt-8 pt-6 border-t border-gray-700"
        >
          <button
            onClick={() => {
              const prevWeek = currentWeek > 1 ? currentWeek - 1 : 12;
              const prevPhase = currentWeek > 1 ? currentPhase : Math.max(1, currentPhase - 1);
              navigate(`/learning/${prevPhase}/${prevWeek}`);
            }}
            disabled={currentPhase === 1 && currentWeek === 1}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg transition-colors disabled:cursor-not-allowed"
          >
            <ArrowLeft className="w-4 h-4" />
            Previous Week
          </button>

          <div className="text-center">
            <div className="text-sm text-gray-400">Week Progress</div>
            <div className="text-lg font-semibold text-white">
              {currentWeek} of 12
            </div>
          </div>

          <button
            onClick={() => {
              const nextWeek = currentWeek < 12 ? currentWeek + 1 : 1;
              const nextPhase = currentWeek < 12 ? currentPhase : Math.min(4, currentPhase + 1);
              navigate(`/learning/${nextPhase}/${nextWeek}`);
            }}
            disabled={currentPhase === 4 && currentWeek === 12}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-lg transition-colors disabled:cursor-not-allowed"
          >
            Next Week
            <ArrowRight className="w-4 h-4" />
          </button>
        </motion.div>
      </div>

      {/* New Vault Unlock Modal */}
      {newVaultUnlock && (
        <VaultRevealModal
          vaultItem={newVaultUnlock}
          isOpen={true}
          onClose={() => setNewVaultUnlock(null)}
          isNewUnlock={true}
        />
      )}
    </motion.div>
  );
};

export default LearningSegment;s