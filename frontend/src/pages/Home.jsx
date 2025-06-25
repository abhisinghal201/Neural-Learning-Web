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

  // FIXED: Use correct API method with date parameter
  const { data: todaySessions } = useQuery(
    'todaySessions',
    () => api.learning.getSessions({ date: new Date().toISOString().split('T')[0] }),
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

  // FIXED: Use correct spaced repetition API method
  const { data: reviewItems } = useQuery(
    'reviewItems',
    () => api.learning.getSpacedRepetition(),
    { refetchInterval: 60000 }
  );

  // Get current streak and study stats
  const currentStreak = progressData?.data?.profile?.current_streak_days || 0;
  const totalStudyTime = progressData?.data?.profile?.total_study_minutes || 0;
  const currentPhase = progressData?.data?.profile?.current_phase || 1;
  const currentWeek = progressData?.data?.profile?.current_week || 1;

  // Get today's session info - FIXED: Updated to match new API response structure
  const todayInfo = todaySessions?.data?.todayStats || {};
  const recommendedSessionType = todaySessions?.data?.recommendations?.nextSessionType || 'math';

  // Format time helper
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  // Get session type info
  const getSessionTypeInfo = (type) => {
    const sessionTypes = {
      math: {
        label: 'Mathematical Theory',
        icon: Brain,
        color: 'text-blue-400',
        bg: 'bg-blue-500/20'
      },
      coding: {
        label: 'Coding Practice',
        icon: Code,
        color: 'text-green-400',
        bg: 'bg-green-500/20'
      },
      visual_projects: {
        label: 'Visual Projects',
        icon: Eye,
        color: 'text-purple-400',
        bg: 'bg-purple-500/20'
      },
      real_applications: {
        label: 'Real Applications',
        icon: Target,
        color: 'text-orange-400',
        bg: 'bg-orange-500/20'
      }
    };
    return sessionTypes[type] || sessionTypes.math;
  };

  // Handle session start
  const handleStartSession = (sessionType) => {
    navigate(`/learning/${currentPhase}/${currentWeek}?session=${sessionType}`);
  };

  // Loading state
  if (progressLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"
          />
          <p className="text-gray-400">Loading your neural dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen bg-gray-900 text-white p-6"
    >
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Neural Odyssey
          </h1>
          <p className="text-gray-400 text-lg">
            Your personal machine learning mastery journey
          </p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
        >
          {/* Current Streak */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-white">{currentStreak}</p>
                <p className="text-sm text-gray-400">Day Streak</p>
              </div>
              <div className="p-3 bg-orange-500/20 rounded-lg">
                <Flame className="w-6 h-6 text-orange-400" />
              </div>
            </div>
          </div>

          {/* Total Study Time */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-white">{formatTime(totalStudyTime)}</p>
                <p className="text-sm text-gray-400">Total Study</p>
              </div>
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Clock className="w-6 h-6 text-blue-400" />
              </div>
            </div>
          </div>

          {/* Current Phase */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-white">Phase {currentPhase}</p>
                <p className="text-sm text-gray-400">Week {currentWeek}</p>
              </div>
              <div className="p-3 bg-green-500/20 rounded-lg">
                <Map className="w-6 h-6 text-green-400" />
              </div>
            </div>
          </div>

          {/* Review Items */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-white">
                  {reviewItems?.data?.todayCount || 0}
                </p>
                <p className="text-sm text-gray-400">Items to Review</p>
              </div>
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <Brain className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Progress and Sessions */}
          <div className="lg:col-span-2 space-y-6">
            {/* Today's Focus */}
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
                  
                  {todayInfo.total_time > 0 && (
                    <div className="text-sm text-gray-400">
                      Already studied {formatTime(todayInfo.total_time)} today
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
                            <span className="text-sm text-gray-300">
                              {getSessionTypeInfo(session.session_type).label}
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

            {/* Progress Visualization */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">Learning Journey</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setActiveView('progress')}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      activeView === 'progress' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    Progress
                  </button>
                  <button
                    onClick={() => setActiveView('skills')}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      activeView === 'skills' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    Skills
                  </button>
                </div>
              </div>

              <div className="h-96">
                <AnimatePresence mode="wait">
                  {activeView === 'progress' && (
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
          </div>

          {/* Right Column - Quick Actions and Review */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={() => navigate(`/learning/${currentPhase}/${currentWeek}`)}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <BookOpen className="w-4 h-4" />
                  Continue Learning
                </button>
                <button
                  onClick={() => navigate('/quests')}
                  className="w-full bg-green-600 hover:bg-green-700 text-white p-3 rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <Target className="w-4 h-4" />
                  Browse Quests
                </button>
                <button
                  onClick={() => navigate('/vault')}
                  className="w-full bg-purple-600 hover:bg-purple-700 text-white p-3 rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Explore Vault
                </button>
              </div>
            </motion.div>

            {/* Review Section */}
            {reviewItems?.data?.reviewItems && reviewItems.data.reviewItems.length > 0 && (
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Spaced Review</h3>
                  <span className="bg-purple-600 text-white text-xs px-2 py-1 rounded-full">
                    {reviewItems.data.todayCount} due
                  </span>
                </div>
                <div className="space-y-3">
                  {reviewItems.data.reviewItems.slice(0, 3).map((item, index) => (
                    <div key={index} className="p-3 bg-gray-700 rounded-lg">
                      <div className="text-sm font-medium text-white mb-1">
                        {item.lesson_title || item.concept_id}
                      </div>
                      <div className="text-xs text-gray-400">
                        {item.lesson_type} • Phase {item.phase}
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => navigate('/review')}
                  className="w-full mt-4 bg-purple-600 hover:bg-purple-700 text-white p-2 rounded-lg text-sm font-medium transition-colors"
                >
                  Start Review Session
                </button>
              </motion.div>
            )}

            {/* Recent Vault Discoveries */}
            {vaultData?.data?.items && (
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Recent Discoveries</h3>
                  <button
                    onClick={() => navigate('/vault')}
                    className="text-purple-400 hover:text-purple-300 text-sm"
                  >
                    View All
                  </button>
                </div>
                <div className="space-y-3">
                  {vaultData.data.slice(0, 3).map((item, index) => (
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