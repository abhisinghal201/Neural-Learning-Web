/**
 * Neural Odyssey Quest Board Component
 *
 * Comprehensive quest management interface providing access to coding challenges,
 * implementation projects, theory quizzes, and practical applications across all
 * phases of the Machine Learning curriculum.
 *
 * Features:
 * - Multi-view quest display (grid, list, kanban)
 * - Advanced filtering and search capabilities
 * - Quest recommendation system
 * - Progress tracking and analytics
 * - Difficulty-based categorization
 * - Real-time submission and feedback
 * - Integration with gamification system
 * - Achievement and milestone tracking
 * - Code playground integration
 * - Collaborative features for peer learning
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Search,
  Filter,
  Grid,
  List,
  Kanban,
  Trophy,
  Target,
  Clock,
  Star,
  BookOpen,
  Code,
  Eye,
  Brain,
  Zap,
  Award,
  TrendingUp,
  Calendar,
  Users,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  CheckCircle,
  Circle,
  Lock,
  Unlock,
  Flame,
  Lightbulb,
  ArrowRight,
  Plus,
  RefreshCw,
  Download,
  Upload,
  Share2,
  Bookmark,
  Settings,
  HelpCircle,
  AlertCircle,
  Info,
  Sparkles,
  Layers,
  Globe,
  Timer,
  BarChart3,
  Compass,
  Map,
  FileText,
  GitBranch,
  Coffee,
  Rocket,
  Shield,
  Cpu,
  Database,
  Terminal,
  Activity
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import QuestCard from '../components/QuestCard';
import CodePlayground from '../components/CodePlayground';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const QuestBoard = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // State management
  const [viewMode, setViewMode] = useState(searchParams.get('view') || 'grid');
  const [selectedCategory, setSelectedCategory] = useState(searchParams.get('category') || 'all');
  const [selectedDifficulty, setSelectedDifficulty] = useState(searchParams.get('difficulty') || 'all');
  const [selectedPhase, setSelectedPhase] = useState(parseInt(searchParams.get('phase')) || 0);
  const [selectedStatus, setSelectedStatus] = useState(searchParams.get('status') || 'all');
  const [searchQuery, setSearchQuery] = useState(searchParams.get('search') || '');
  const [sortBy, setSortBy] = useState(searchParams.get('sort') || 'recommended');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedQuest, setSelectedQuest] = useState(null);
  const [showCodePlayground, setShowCodePlayground] = useState(false);
  const [questSubmissionMode, setQuestSubmissionMode] = useState(false);

  // Refs
  const searchInputRef = useRef(null);

  // Fetch quests data
  const { data: questsData, isLoading: questsLoading, refetch: refetchQuests } = useQuery(
    ['quests', selectedCategory, selectedDifficulty, selectedPhase, selectedStatus, sortBy],
    () => api.get('/learning/quests', {
      params: {
        type: selectedCategory !== 'all' ? selectedCategory : undefined,
        difficulty: selectedDifficulty !== 'all' ? selectedDifficulty : undefined,
        phase: selectedPhase > 0 ? selectedPhase : undefined,
        status: selectedStatus !== 'all' ? selectedStatus : undefined,
        sort: sortBy,
        limit: 100
      }
    }),
    {
      refetchInterval: 30000,
      staleTime: 15000
    }
  );

  // Fetch user progress
  const { data: progressData } = useQuery(
    'learningProgress',
    () => api.get('/learning/progress'),
    {
      refetchInterval: 60000
    }
  );

  // Fetch quest recommendations
  const { data: recommendationsData } = useQuery(
    'questRecommendations',
    () => api.get('/learning/quest-recommendations'),
    {
      refetchInterval: 120000
    }
  );

  // Fetch quest analytics
  const { data: analyticsData } = useQuery(
    'questAnalytics',
    () => api.get('/learning/quest-analytics'),
    {
      refetchInterval: 300000
    }
  );

  // Quest submission mutation
  const submitQuestMutation = useMutation(
    (questData) => api.post('/learning/quests', questData),
    {
      onSuccess: (data) => {
        toast.success('Quest submitted successfully!');
        queryClient.invalidateQueries(['quests']);
        queryClient.invalidateQueries(['learningProgress']);
        queryClient.invalidateQueries(['questAnalytics']);
        setQuestSubmissionMode(false);
        setShowCodePlayground(false);
      },
      onError: (error) => {
        toast.error(error.response?.data?.message || 'Failed to submit quest');
      }
    }
  );

  // Quest categories configuration
  const questCategories = [
    {
      id: 'all',
      label: 'All Quests',
      icon: Globe,
      color: 'gray',
      description: 'View all available quests'
    },
    {
      id: 'coding_exercise',
      label: 'Coding Exercises',
      icon: Code,
      color: 'green',
      description: 'Hands-on programming challenges'
    },
    {
      id: 'implementation_project',
      label: 'Implementation Projects',
      icon: Layers,
      color: 'blue',
      description: 'Complex multi-part projects'
    },
    {
      id: 'theory_quiz',
      label: 'Theory Quizzes',
      icon: BookOpen,
      color: 'purple',
      description: 'Knowledge validation assessments'
    },
    {
      id: 'practical_application',
      label: 'Practical Applications',
      icon: Target,
      color: 'orange',
      description: 'Real-world problem solving'
    }
  ];

  // Difficulty levels
  const difficultyLevels = [
    { id: 'all', label: 'All Levels', color: 'gray' },
    { id: 'beginner', label: 'Beginner', color: 'green' },
    { id: 'intermediate', label: 'Intermediate', color: 'yellow' },
    { id: 'advanced', label: 'Advanced', color: 'orange' },
    { id: 'expert', label: 'Expert', color: 'red' }
  ];

  // Status options
  const statusOptions = [
    { id: 'all', label: 'All Status', icon: Globe },
    { id: 'available', label: 'Available', icon: Unlock },
    { id: 'locked', label: 'Locked', icon: Lock },
    { id: 'in_progress', label: 'In Progress', icon: Play },
    { id: 'attempted', label: 'Attempted', icon: Circle },
    { id: 'completed', label: 'Completed', icon: CheckCircle },
    { id: 'mastered', label: 'Mastered', icon: Trophy }
  ];

  // Sort options
  const sortOptions = [
    { id: 'recommended', label: 'Recommended' },
    { id: 'difficulty_asc', label: 'Difficulty: Easy First' },
    { id: 'difficulty_desc', label: 'Difficulty: Hard First' },
    { id: 'phase_asc', label: 'Phase: 1 to 4' },
    { id: 'recent', label: 'Recently Added' },
    { id: 'popular', label: 'Most Popular' },
    { id: 'completion_rate', label: 'Completion Rate' }
  ];

  // Learning phases
  const learningPhases = [
    { id: 0, label: 'All Phases', color: 'gray' },
    { id: 1, label: 'Phase 1: Foundations', color: 'blue' },
    { id: 2, label: 'Phase 2: Core ML', color: 'green' },
    { id: 3, label: 'Phase 3: Advanced AI', color: 'purple' },
    { id: 4, label: 'Phase 4: Mastery', color: 'orange' }
  ];

  // Filter and search quests
  const filteredQuests = useMemo(() => {
    let quests = questsData?.data?.quests || [];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      quests = quests.filter(quest =>
        quest.title.toLowerCase().includes(query) ||
        quest.description.toLowerCase().includes(query) ||
        quest.tags?.some(tag => tag.toLowerCase().includes(query))
      );
    }

    return quests;
  }, [questsData, searchQuery]);

  // Calculate quest statistics
  const questStats = useMemo(() => {
    const quests = questsData?.data?.quests || [];
    
    const totalQuests = quests.length;
    const completedQuests = quests.filter(q => q.status === 'completed' || q.status === 'mastered').length;
    const masteredQuests = quests.filter(q => q.status === 'mastered').length;
    const availableQuests = quests.filter(q => q.status === 'available').length;
    const inProgressQuests = quests.filter(q => q.status === 'in_progress').length;

    const categoryStats = questCategories.slice(1).map(category => ({
      ...category,
      count: quests.filter(q => q.type === category.id).length,
      completed: quests.filter(q => q.type === category.id && (q.status === 'completed' || q.status === 'mastered')).length
    }));

    return {
      totalQuests,
      completedQuests,
      masteredQuests,
      availableQuests,
      inProgressQuests,
      completionRate: totalQuests > 0 ? (completedQuests / totalQuests) * 100 : 0,
      categoryStats
    };
  }, [questsData, questCategories]);

  // Handle filter changes
  const updateFilters = (key, value) => {
    const newParams = new URLSearchParams(searchParams);
    if (value === 'all' || value === 0 || value === '') {
      newParams.delete(key);
    } else {
      newParams.set(key, value.toString());
    }
    setSearchParams(newParams);
  };

  // Handle quest selection
  const handleQuestSelect = (quest) => {
    setSelectedQuest(quest);
    if (quest.type === 'coding_exercise' || quest.type === 'implementation_project') {
      setShowCodePlayground(true);
    } else {
      navigate(`/quests/${quest.id}`);
    }
  };

  // Handle quest submission
  const handleQuestSubmission = async (questData) => {
    try {
      await submitQuestMutation.mutateAsync({
        quest_id: selectedQuest.id,
        quest_title: selectedQuest.title,
        quest_type: selectedQuest.type,
        phase: selectedQuest.phase,
        week: selectedQuest.week,
        status: questData.status,
        time_to_complete_minutes: questData.timeSpent,
        user_code: questData.code,
        final_solution: questData.solution,
        test_results: questData.testResults,
        difficulty_rating: questData.difficultyRating,
        notes: questData.notes
      });
    } catch (error) {
      console.error('Quest submission failed:', error);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        searchInputRef.current?.focus();
      }
      if (event.key === 'Escape') {
        setShowCodePlayground(false);
        setSelectedQuest(null);
      }
    };

    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, []);

  // Update URL params when filters change
  useEffect(() => {
    updateFilters('view', viewMode);
  }, [viewMode]);

  useEffect(() => {
    updateFilters('category', selectedCategory);
  }, [selectedCategory]);

  useEffect(() => {
    updateFilters('difficulty', selectedDifficulty);
  }, [selectedDifficulty]);

  useEffect(() => {
    updateFilters('phase', selectedPhase);
  }, [selectedPhase]);

  useEffect(() => {
    updateFilters('status', selectedStatus);
  }, [selectedStatus]);

  useEffect(() => {
    updateFilters('search', searchQuery);
  }, [searchQuery]);

  useEffect(() => {
    updateFilters('sort', sortBy);
  }, [sortBy]);

  // Render quest grid
  const renderQuestGrid = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {filteredQuests.map((quest) => (
        <motion.div
          key={quest.id}
          layout
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.2 }}
        >
          <QuestCard
            quest={quest}
            onStart={() => handleQuestSelect(quest)}
            className="h-full"
          />
        </motion.div>
      ))}
    </div>
  );

  // Render quest list
  const renderQuestList = () => (
    <div className="space-y-4">
      {filteredQuests.map((quest) => {
        const categoryInfo = questCategories.find(c => c.id === quest.type);
        const difficultyInfo = difficultyLevels.find(d => d.id === quest.difficulty);

        return (
          <motion.div
            key={quest.id}
            layout
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-colors cursor-pointer"
            onClick={() => handleQuestSelect(quest)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1">
                <div className={`p-3 bg-${categoryInfo?.color}-600 rounded-lg`}>
                  <categoryInfo.icon className="w-6 h-6 text-white" />
                </div>

                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className="text-lg font-semibold text-white">{quest.title}</h3>
                    <span className={`px-2 py-1 bg-${difficultyInfo?.color}-600 text-white text-xs rounded-full`}>
                      {difficultyInfo?.label}
                    </span>
                    <span className="px-2 py-1 bg-gray-600 text-white text-xs rounded-full">
                      Phase {quest.phase}
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm line-clamp-2">{quest.description}</p>
                  
                  <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                    <div className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{quest.estimated_time_minutes || 30}m</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Target className="w-3 h-3" />
                      <span>{quest.learning_objectives?.length || 0} objectives</span>
                    </div>
                    {quest.tags && quest.tags.length > 0 && (
                      <div className="flex items-center space-x-1">
                        <span>{quest.tags.slice(0, 2).join(', ')}</span>
                        {quest.tags.length > 2 && <span>+{quest.tags.length - 2}</span>}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                {/* Progress indicator */}
                <div className="text-right">
                  <div className="text-sm font-medium text-white capitalize">{quest.status || 'available'}</div>
                  {quest.completion_percentage > 0 && (
                    <div className="text-xs text-gray-400">{quest.completion_percentage}% complete</div>
                  )}
                </div>

                <ArrowRight className="w-5 h-5 text-gray-400" />
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );

  // Render kanban view
  const renderKanbanView = () => {
    const kanbanColumns = [
      { id: 'available', title: 'Available', color: 'blue' },
      { id: 'in_progress', title: 'In Progress', color: 'yellow' },
      { id: 'completed', title: 'Completed', color: 'green' },
      { id: 'mastered', title: 'Mastered', color: 'purple' }
    ];

    return (
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {kanbanColumns.map((column) => {
          const columnQuests = filteredQuests.filter(quest => quest.status === column.id);

          return (
            <div key={column.id} className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className={`font-semibold text-${column.color}-400`}>{column.title}</h3>
                <span className="bg-gray-700 text-white text-xs px-2 py-1 rounded-full">
                  {columnQuests.length}
                </span>
              </div>

              <div className="space-y-3 max-h-96 overflow-y-auto">
                {columnQuests.map((quest) => {
                  const categoryInfo = questCategories.find(c => c.id === quest.type);

                  return (
                    <motion.div
                      key={quest.id}
                      layout
                      className="bg-gray-700 border border-gray-600 rounded-lg p-3 cursor-pointer hover:border-blue-500 transition-colors"
                      onClick={() => handleQuestSelect(quest)}
                    >
                      <div className="flex items-center space-x-2 mb-2">
                        <categoryInfo.icon className="w-4 h-4 text-gray-400" />
                        <span className="text-sm font-medium text-white line-clamp-1">{quest.title}</span>
                      </div>
                      <p className="text-xs text-gray-400 line-clamp-2 mb-2">{quest.description}</p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Phase {quest.phase}</span>
                        <span>{quest.estimated_time_minutes || 30}m</span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render loading state
  if (questsLoading) {
    return (
      <div className="quest-board h-full">
        <LoadingSpinner
          size="large"
          text="Loading quest board..."
          description="Preparing your personalized quest recommendations"
        />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="quest-board min-h-screen bg-gray-900"
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
              <h1 className="text-4xl font-bold text-white mb-2">Quest Board</h1>
              <p className="text-gray-400 text-lg">
                Challenge yourself with hands-on projects and exercises
              </p>
            </div>

            <div className="flex items-center space-x-4">
              {/* Quest Statistics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-center">
                  <div className="text-2xl font-bold text-blue-400">{questStats.totalQuests}</div>
                  <div className="text-xs text-gray-400">Total Quests</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-center">
                  <div className="text-2xl font-bold text-green-400">{questStats.completedQuests}</div>
                  <div className="text-xs text-gray-400">Completed</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-center">
                  <div className="text-2xl font-bold text-purple-400">{questStats.masteredQuests}</div>
                  <div className="text-xs text-gray-400">Mastered</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-center">
                  <div className="text-2xl font-bold text-orange-400">{Math.round(questStats.completionRate)}%</div>
                  <div className="text-xs text-gray-400">Success Rate</div>
                </div>
              </div>
            </div>
          </div>

          {/* Recommended Quests */}
          {recommendationsData?.data?.recommendedQuests && (
            <div className="bg-gradient-to-r from-blue-900 to-purple-900 border border-blue-700 rounded-lg p-6 mb-6">
              <div className="flex items-center space-x-2 mb-4">
                <Sparkles className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">Recommended for You</h3>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {recommendationsData.data.recommendedQuests.slice(0, 3).map((quest) => (
                  <div
                    key={quest.id}
                    className="bg-white bg-opacity-10 rounded-lg p-4 cursor-pointer hover:bg-opacity-20 transition-all"
                    onClick={() => handleQuestSelect(quest)}
                  >
                    <h4 className="font-medium text-white mb-1">{quest.title}</h4>
                    <p className="text-sm text-gray-300 mb-2">{quest.reason}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">{quest.estimated_time_minutes}m</span>
                      <ArrowRight className="w-4 h-4 text-blue-400" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Filters and Controls */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="mb-6"
        >
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
            {/* Search and View Controls */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4 flex-1">
                {/* Search */}
                <div className="relative flex-1 max-w-md">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    ref={searchInputRef}
                    type="text"
                    placeholder="Search quests... (Ctrl+K)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none"
                  />
                </div>

                {/* Filter Toggle */}
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                    showFilters ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <Filter className="w-4 h-4" />
                  <span>Filters</span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                </button>
              </div>

              {/* View Mode Controls */}
              <div className="flex items-center space-x-2">
                <div className="flex bg-gray-700 rounded-lg p-1">
                  {[
                    { id: 'grid', icon: Grid, label: 'Grid' },
                    { id: 'list', icon: List, label: 'List' },
                    { id: 'kanban', icon: Kanban, label: 'Kanban' }
                  ].map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => setViewMode(mode.id)}
                      className={`p-2 rounded transition-colors ${
                        viewMode === mode.id ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
                      }`}
                      title={mode.label}
                    >
                      <mode.icon className="w-4 h-4" />
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => refetchQuests()}
                  className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-gray-400 hover:text-white"
                  title="Refresh"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Expandable Filters */}
            <AnimatePresence>
              {showFilters && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden border-t border-gray-700 pt-4"
                >
                  <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                    {/* Category Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Category</label>
                      <select
                        value={selectedCategory}
                        onChange={(e) => setSelectedCategory(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                      >
                        {questCategories.map((category) => (
                          <option key={category.id} value={category.id}>
                            {category.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Difficulty Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Difficulty</label>
                      <select
                        value={selectedDifficulty}
                        onChange={(e) => setSelectedDifficulty(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                      >
                        {difficultyLevels.map((level) => (
                          <option key={level.id} value={level.id}>
                            {level.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Phase Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Phase</label>
                      <select
                        value={selectedPhase}
                        onChange={(e) => setSelectedPhase(parseInt(e.target.value))}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                      >
                        {learningPhases.map((phase) => (
                          <option key={phase.id} value={phase.id}>
                            {phase.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Status Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Status</label>
                      <select
                        value={selectedStatus}
                        onChange={(e) => setSelectedStatus(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                      >
                        {statusOptions.map((status) => (
                          <option key={status.id} value={status.id}>
                            {status.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Sort Options */}
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Sort By</label>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                      >
                        {sortOptions.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Category Quick Filters */}
                  <div className="mt-4">
                    <div className="flex items-center space-x-2 flex-wrap">
                      {questCategories.slice(1).map((category) => {
                        const count = questStats.categoryStats.find(s => s.id === category.id)?.count || 0;
                        return (
                          <button
                            key={category.id}
                            onClick={() => setSelectedCategory(category.id)}
                            className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                              selectedCategory === category.id
                                ? `bg-${category.color}-600 text-white`
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                          >
                            <category.icon className="w-4 h-4" />
                            <span>{category.label}</span>
                            <span className="bg-black bg-opacity-20 px-2 py-1 rounded text-xs">{count}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Results Summary */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <div className="flex items-center justify-between text-gray-400">
            <div>
              Showing {filteredQuests.length} of {questStats.totalQuests} quests
              {searchQuery && (
                <span> matching "{searchQuery}"</span>
              )}
            </div>
            {filteredQuests.length === 0 && questStats.totalQuests > 0 && (
              <div className="text-orange-400">
                No quests found. Try adjusting your filters.
              </div>
            )}
          </div>
        </motion.div>

        {/* Quest Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {filteredQuests.length > 0 ? (
            <AnimatePresence mode="wait">
              {viewMode === 'grid' && (
                <motion.div key="grid" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  {renderQuestGrid()}
                </motion.div>
              )}
              {viewMode === 'list' && (
                <motion.div key="list" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  {renderQuestList()}
                </motion.div>
              )}
              {viewMode === 'kanban' && (
                <motion.div key="kanban" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  {renderKanbanView()}
                </motion.div>
              )}
            </AnimatePresence>
          ) : (
            <div className="text-center py-12">
              <div className="w-24 h-24 mx-auto mb-6 bg-gray-800 rounded-full flex items-center justify-center">
                <Target className="w-12 h-12 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {questStats.totalQuests === 0 ? 'No Quests Available' : 'No Matching Quests'}
              </h3>
              <p className="text-gray-400 mb-6">
                {questStats.totalQuests === 0
                  ? 'Quests will become available as you progress through the learning path.'
                  : 'Try adjusting your search criteria or filters to find more quests.'
                }
              </p>
              {questStats.totalQuests === 0 && (
                <button
                  onClick={() => navigate('/learning')}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center space-x-2 mx-auto"
                >
                  <BookOpen className="w-5 h-5" />
                  <span>Start Learning Path</span>
                </button>
              )}
            </div>
          )}
        </motion.div>

        {/* Code Playground Modal */}
        <AnimatePresence>
          {showCodePlayground && selectedQuest && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-black bg-opacity-75 flex items-center justify-center p-4"
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="w-full max-w-6xl h-[90vh] bg-gray-900 rounded-lg overflow-hidden border border-gray-700"
              >
                <div className="h-full flex flex-col">
                  {/* Header */}
                  <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{selectedQuest.title}</h3>
                      <p className="text-sm text-gray-400">{selectedQuest.type.replace('_', ' ')}</p>
                    </div>
                    <button
                      onClick={() => {
                        setShowCodePlayground(false);
                        setSelectedQuest(null);
                      }}
                      className="p-2 hover:bg-gray-800 rounded-lg transition-colors text-gray-400 hover:text-white"
                    >
                      ×
                    </button>
                  </div>

                  {/* Code Playground */}
                  <div className="flex-1">
                    <CodePlayground
                      quest={selectedQuest}
                      onSubmit={handleQuestSubmission}
                      onClose={() => {
                        setShowCodePlayground(false);
                        setSelectedQuest(null);
                      }}
                    />
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default QuestBoard;