import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
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
  Sparkles
} from 'lucide-react';

// Components
import ProgressMap from '../components/Dashboard/ProgressMap';
import SkillTree from '../components/Dashboard/SkillTree';
import QuestCard from '../components/QuestCard';
import VaultRevealModal from '../components/VaultRevealModal';

// Utils
import { api } from '../utils/api';

const Home = () => {
  const [activeView, setActiveView] = useState('dashboard');
  const [selectedVaultItem, setSelectedVaultItem] = useState(null);
  const [showVaultModal, setShowVaultModal] = useState(false);
  const navigate = useNavigate();

  // Fetch dashboard data
  const { data: progressData, isLoading: progressLoading } = useQuery(
    'learningProgress',
    () => api.get('/learning/progress'),
    { refetchInterval: 30000 }
  );

  const { data: todaySessions } = useQuery(
    'todaySessions',
    () => api.get('/learning/sessions/today'),
    { refetchInterval: 60000 }
  );

  const { data: analyticsData } = useQuery(
    'analytics',
    () => api.get('/analytics/summary'),
    { refetchInterval: 60000 }
  );

  const { data: vaultData } = useQuery(
    'vaultItems',
    () => api.get('/vault/items'),
    { refetchInterval: 60000 }
  );

  const { data: recommendedQuests } = useQuery(
    'recommendedQuests',
    () => api.get('/learning/quests?limit=3'),
    { refetchInterval: 60000 }
  );

  const { data: reviewItems } = useQuery(
    'reviewItems',
    () => api.get('/learning/review'),
    { refetchInterval: 60000 }
  );

  // Get current streak and study stats
  const currentStreak = progressData?.data?.profile?.current_streak_days || 0;
  const totalStudyTime = progressData?.data?.profile?.total_study_minutes || 0;
  const currentPhase = progressData?.data?.profile?.current_phase || 1;
  const currentWeek = progressData?.data?.profile?.current_week || 1;

  // Get today's session info
  const todayInfo = todaySessions?.data?.summary || {};
  const recommendedSessionType = todayInfo.recommended_next_type || 'math';

  // Format time helper
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  // Get session type info
  const getSessionTypeInfo = (type) => {
    const types = {
      math: { icon: Brain, label: 'Mathematics', color: 'text-blue-400', bg: 'bg-blue-400/10' },
      coding: { icon: Code, label: 'Coding Practice', color: 'text-green-400', bg: 'bg-green-400/10' },
      visual_projects: { icon: Eye, label: 'Visual Projects', color: 'text-purple-400', bg: 'bg-purple-400/10' },
      real_applications: { icon: Target, label: 'Real Applications', color: 'text-orange-400', bg: 'bg-orange-400/10' }
    };
    return types[type] || types.math;
  };

  // Handle vault item click
  const handleVaultItemClick = (item) => {
    if (item.unlocked) {
      setSelectedVaultItem(item);
      setShowVaultModal(true);
    }
  };

  // Start learning session
  const handleStartSession = (type) => {
    navigate(`/learning?session=${type}`);
  };

  if (progressLoading) {
    return (
      <div className="home-page h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-blue-400">
          <Brain className="w-8 h-8 animate-pulse" />
          <span className="text-xl">Loading your Neural Odyssey...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="home-page h-full overflow-auto"
    >
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Welcome Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-2">
            Welcome back, <span className="text-blue-400">Neural Explorer</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Continue your journey through the fascinating world of machine learning
          </p>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
        >
          {/* Current Progress */}
          <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/30 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <Map className="w-5 h-5 text-blue-400" />
              <span className="text-2xl font-bold text-white">
                {currentPhase}.{currentWeek}
              </span>
            </div>
            <h3 className="font-semibold text-white mb-1">Current Progress</h3>
            <p className="text-blue-400 text-sm">Phase {currentPhase}, Week {currentWeek}</p>
          </div>

          {/* Study Streak */}
          <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <Flame className="w-5 h-5 text-orange-400" />
              <span className="text-2xl font-bold text-white">{currentStreak}</span>
            </div>
            <h3 className="font-semibold text-white mb-1">Study Streak</h3>
            <p className="text-orange-400 text-sm">
              {currentStreak > 0 ? 'Keep it going!' : 'Start your streak today'}
            </p>
          </div>

          {/* Total Study Time */}
          <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <Clock className="w-5 h-5 text-green-400" />
              <span className="text-2xl font-bold text-white">{formatTime(totalStudyTime)}</span>
            </div>
            <h3 className="font-semibold text-white mb-1">Total Study Time</h3>
            <p className="text-green-400 text-sm">Lifetime learning</p>
          </div>

          {/* Vault Unlocks */}
          <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              <span className="text-2xl font-bold text-white">
                {vaultData?.data?.statistics?.unlocked_items || 0}
              </span>
            </div>
            <h3 className="font-semibold text-white mb-1">Vault Unlocks</h3>
            <p className="text-purple-400 text-sm">Secrets discovered</p>
          </div>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Today's Focus */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {/* Today's Recommended Session */}
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-blue-400" />
                  Today's Focus
                </h2>
                <span className="text-sm text-gray-400">
                  {new Date().toLocaleDateString('en-US', { 
                    weekday: 'long', 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </span>
              </div>

              <div className="space-y-4">
                {/* Recommended Session */}
                <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {(() => {
                        const sessionInfo = getSessionTypeInfo(recommendedSessionType);
                        const IconComponent = sessionInfo.icon;
                        return (
                          <>
                            <div className={`p-2 rounded-lg ${sessionInfo.bg}`}>
                              <IconComponent className={`w-5 h-5 ${sessionInfo.color}`} />
                            </div>
                            <div>
                              <h3 className="font-semibold text-white">{sessionInfo.label}</h3>
                              <p className="text-sm text-gray-400">Recommended for today</p>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleStartSession(recommendedSessionType)}
                      className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Start
                    </motion.button>
                  </div>
                  
                  {todayInfo.total_time_minutes > 0 && (
                    <div className="text-sm text-gray-400">
                      Already studied {formatTime(todayInfo.total_time_minutes)} today
                    </div>
                  )}
                </div>

                {/* Today's Sessions */}
                {todaySessions?.data?.sessions && todaySessions.data.sessions.length > 0 && (
                  <div>
                    <h4 className="font-medium text-white mb-2">Today's Sessions</h4>
                    <div className="space-y-2">
                      {todaySessions.data.sessions.slice(0, 3).map((session, index) => (
                        <div key={index} className="flex items-center justify-between py-2 px-3 bg-gray-900/30 rounded-lg">
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${
                              session.session_type === 'math' ? 'bg-blue-400' :
                              session.session_type === 'coding' ? 'bg-green-400' :
                              session.session_type === 'visual_projects' ? 'bg-purple-400' : 'bg-orange-400'
                            }`} />
                            <span className="text-sm text-gray-300 capitalize">
                              {session.session_type.replace('_', ' ')}
                            </span>
                          </div>
                          <span className="text-sm text-gray-400">
                            {formatTime(session.duration_minutes || 0)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Spaced Repetition Review */}
            {reviewItems?.data?.review_items && reviewItems.data.review_items.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Lightbulb className="w-5 h-5 text-yellow-400" />
                    Review Due
                  </h2>
                  <span className="bg-yellow-400/20 text-yellow-400 px-2 py-1 rounded-full text-sm font-medium">
                    {reviewItems.data.review_items.length}
                  </span>
                </div>

                <div className="space-y-2">
                  {reviewItems.data.review_items.slice(0, 3).map((item, index) => (
                    <div key={index} className="flex items-center justify-between py-2 px-3 bg-gray-900/30 rounded-lg">
                      <span className="text-sm text-gray-300">{item.concept_title}</span>
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
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => navigate('/learning?tab=review')}
                  className="w-full mt-4 bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <Zap className="w-4 h-4" />
                  Start Review Session
                </button>
              </div>
            )}

            {/* Recent Vault Unlocks */}
            {vaultData?.data?.items && (
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Trophy className="w-5 h-5 text-purple-400" />
                    Neural Vault
                  </h2>
                  <button
                    onClick={() => navigate('/vault')}
                    className="text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1 text-sm"
                  >
                    View All
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </div>

                <div className="space-y-3">
                  {Object.values(vaultData.data.items).flat()
                    .filter(item => item.unlocked)
                    .slice(0, 3)
                    .map((item, index) => (
                      <motion.div
                        key={index}
                        whileHover={{ scale: 1.02 }}
                        onClick={() => handleVaultItemClick(item)}
                        className="flex items-center gap-3 p-3 bg-gray-900/30 rounded-lg cursor-pointer hover:bg-gray-900/50 transition-colors"
                      >
                        <span className="text-2xl">{item.icon}</span>
                        <div className="flex-1">
                          <h4 className="font-medium text-white text-sm">{item.title}</h4>
                          <p className="text-xs text-gray-400 capitalize">
                            {item.category.replace('_', ' ')}
                          </p>
                        </div>
                        {item.is_read && (
                          <Eye className="w-4 h-4 text-green-400" />
                        )}
                      </motion.div>
                    ))}
                </div>
              </div>
            )}
          </motion.div>

          {/* Center Column - Main Visualization */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2"
          >
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden">
              {/* View Toggle */}
              <div className="flex items-center justify-between p-4 border-b border-gray-700">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveView('dashboard')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                      activeView === 'dashboard' 
                        ? 'bg-blue-500 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    Progress Map
                  </button>
                  <button
                    onClick={() => setActiveView('skills')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                      activeView === 'skills' 
                        ? 'bg-blue-500 text-white' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Award className="w-4 h-4" />
                    Skill Tree
                  </button>
                </div>
                
                <button
                  onClick={() => navigate('/learning')}
                  className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                >
                  Continue Learning
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>

              {/* Visualization Content */}
              <div className="h-96">
                <AnimatePresence mode="wait">
                  {activeView === 'dashboard' && (
                    <motion.div
                      key="progress"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      className="h-full"
                    >
                      <ProgressMap />
                    </motion.div>
                  )}
                  {activeView === 'skills' && (
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
            </div>

            {/* Recommended Quests */}
            {recommendedQuests?.data && recommendedQuests.data.length > 0 && (
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="mt-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Target className="w-5 h-5 text-green-400" />
                    Recommended Quests
                  </h2>
                  <button
                    onClick={() => navigate('/learning?tab=quests')}
                    className="text-green-400 hover:text-green-300 transition-colors flex items-center gap-1 text-sm"
                  >
                    View All Quests
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {recommendedQuests.data.slice(0, 3).map((quest, index) => (
                    <QuestCard
                      key={quest.id}
                      quest={quest}
                      onStart={() => navigate(`/learning?quest=${quest.id}`)}
                      className="h-fit"
                    />
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>

      {/* Vault Reveal Modal */}
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

export default Home;