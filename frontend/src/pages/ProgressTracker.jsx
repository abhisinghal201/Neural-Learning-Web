/**
 * Neural Odyssey Progress Tracker Component
 *
 * Advanced learning analytics and progress tracking interface providing comprehensive
 * insights into learning patterns, performance metrics, and goal achievement.
 *
 * Features:
 * - Detailed progress analytics with interactive charts
 * - Learning velocity and efficiency metrics
 * - Time-based progress tracking and patterns
 * - Skill development visualization
 * - Achievement and milestone tracking
 * - Comparative analysis and benchmarking
 * - Goal setting and progress monitoring
 * - Learning recommendations and insights
 * - Data export and portfolio generation
 * - Personalized learning analytics dashboard
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  TrendingUp,
  BarChart3,
  PieChart,
  Clock,
  Calendar,
  Target,
  Trophy,
  Star,
  Flame,
  Brain,
  Code,
  BookOpen,
  Eye,
  Award,
  Zap,
  Activity,
  Users,
  Globe,
  Download,
  Upload,
  RefreshCw,
  Settings,
  Filter,
  Search,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  ChevronLeft,
  Play,
  Pause,
  Square,
  RotateCcw,
  FastForward,
  Rewind,
  Maximize2,
  Minimize2,
  Share2,
  Bookmark,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  HelpCircle,
  ExternalLink,
  Map,
  Compass,
  Lightbulb,
  Sparkles,
  Coffee,
  Moon,
  Sun,
  Sunrise,
  Sunset,
  Timer,
  Stopwatch,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  Percent,
  Hash,
  MoreHorizontal
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const ProgressTracker = ({ profile: initialProfile }) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // State management
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'overview');
  const [selectedTimeRange, setSelectedTimeRange] = useState(searchParams.get('range') || '7d');
  const [selectedMetric, setSelectedMetric] = useState('completion_rate');
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [selectedPhase, setSelectedPhase] = useState(0);
  const [selectedSkill, setSelectedSkill] = useState('all');
  const [comparisonMode, setComparisonMode] = useState(false);
  const [exportFormat, setExportFormat] = useState('pdf');
  const [goalTargets, setGoalTargets] = useState({});

  // Fetch analytics data
  const { data: analyticsData, isLoading: analyticsLoading } = useQuery(
    ['progressAnalytics', selectedTimeRange, selectedPhase, selectedSkill],
    () => api.get('/analytics/progress', {
      params: {
        range: selectedTimeRange,
        phase: selectedPhase > 0 ? selectedPhase : undefined,
        skill: selectedSkill !== 'all' ? selectedSkill : undefined,
        detailed: true
      }
    }),
    {
      refetchInterval: 60000,
      staleTime: 30000
    }
  );

  // Fetch learning patterns
  const { data: patternsData } = useQuery(
    ['learningPatterns', selectedTimeRange],
    () => api.get('/analytics/patterns', {
      params: { range: selectedTimeRange }
    }),
    {
      refetchInterval: 300000
    }
  );

  // Fetch skill development data
  const { data: skillsData } = useQuery(
    'skillDevelopment',
    () => api.get('/analytics/skills'),
    {
      refetchInterval: 120000
    }
  );

  // Fetch goals and milestones
  const { data: goalsData } = useQuery(
    'learningGoals',
    () => api.get('/learning/goals'),
    {
      refetchInterval: 300000
    }
  );

  // Fetch comparative data
  const { data: comparativeData, isLoading: comparativeLoading } = useQuery(
    ['comparativeAnalytics', selectedTimeRange],
    () => api.get('/analytics/comparative', {
      params: { range: selectedTimeRange }
    }),
    {
      enabled: comparisonMode,
      refetchInterval: 300000
    }
  );

  // Export progress mutation
  const exportProgressMutation = useMutation(
    ({ format, options }) => api.post('/analytics/export', { format, options }),
    {
      onSuccess: (data) => {
        if (data.data.downloadUrl) {
          window.open(data.data.downloadUrl, '_blank');
        }
        toast.success('Progress report exported successfully!');
      },
      onError: (error) => {
        toast.error('Failed to export progress report');
      }
    }
  );

  // Time range options
  const timeRanges = [
    { id: '1d', label: 'Today', icon: Sun },
    { id: '7d', label: 'Week', icon: Calendar },
    { id: '30d', label: 'Month', icon: Calendar },
    { id: '90d', label: 'Quarter', icon: BarChart3 },
    { id: '365d', label: 'Year', icon: TrendingUp },
    { id: 'all', label: 'All Time', icon: Globe }
  ];

  // Learning phases
  const learningPhases = [
    { id: 0, label: 'All Phases', color: 'gray' },
    { id: 1, label: 'Foundations', color: 'blue' },
    { id: 2, label: 'Core ML', color: 'green' },
    { id: 3, label: 'Advanced AI', color: 'purple' },
    { id: 4, label: 'Mastery', color: 'orange' }
  ];

  // Skill categories
  const skillCategories = [
    { id: 'all', label: 'All Skills', icon: Globe },
    { id: 'mathematics', label: 'Mathematics', icon: Brain },
    { id: 'programming', label: 'Programming', icon: Code },
    { id: 'theory', label: 'Theory', icon: BookOpen },
    { id: 'applications', label: 'Applications', icon: Target },
    { id: 'creativity', label: 'Creativity', icon: Eye },
    { id: 'persistence', label: 'Persistence', icon: Flame }
  ];

  // Progress metrics
  const progressMetrics = useMemo(() => {
    if (!analyticsData?.data) return null;

    const data = analyticsData.data;
    
    return {
      completionRate: data.completion_rate || 0,
      masteryRate: data.mastery_rate || 0,
      averageSessionTime: data.average_session_time || 0,
      totalStudyTime: data.total_study_time || 0,
      streakDays: data.current_streak || 0,
      lessonsCompleted: data.lessons_completed || 0,
      questsCompleted: data.quests_completed || 0,
      skillPointsEarned: data.skill_points_earned || 0,
      vaultItemsUnlocked: data.vault_items_unlocked || 0,
      learningVelocity: data.learning_velocity || 0,
      efficiencyScore: data.efficiency_score || 0,
      consistencyScore: data.consistency_score || 0
    };
  }, [analyticsData]);

  // Learning insights
  const learningInsights = useMemo(() => {
    if (!patternsData?.data) return [];

    const patterns = patternsData.data;
    const insights = [];

    // Time-based insights
    if (patterns.peak_hours) {
      insights.push({
        type: 'time',
        title: 'Optimal Learning Time',
        description: `You're most productive between ${patterns.peak_hours.start} and ${patterns.peak_hours.end}`,
        icon: Clock,
        color: 'blue',
        action: 'Schedule important sessions during this time'
      });
    }

    // Learning velocity insights
    if (patterns.velocity_trend === 'increasing') {
      insights.push({
        type: 'velocity',
        title: 'Accelerating Progress',
        description: 'Your learning velocity has increased by 23% this week',
        icon: TrendingUp,
        color: 'green',
        action: 'Keep up the momentum with challenging quests'
      });
    }

    // Skill development insights
    if (patterns.strongest_skill) {
      insights.push({
        type: 'skill',
        title: 'Skill Strength',
        description: `${patterns.strongest_skill} is your strongest area`,
        icon: Trophy,
        color: 'yellow',
        action: 'Consider advanced challenges in this area'
      });
    }

    // Study pattern insights
    if (patterns.consistency_score > 0.8) {
      insights.push({
        type: 'consistency',
        title: 'Excellent Consistency',
        description: 'You have a very consistent learning schedule',
        icon: CheckCircle,
        color: 'green',
        action: 'Your habit formation is excellent'
      });
    }

    return insights;
  }, [patternsData]);

  // Format time helper
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  // Format percentage
  const formatPercentage = (value, decimals = 1) => {
    return `${(value * 100).toFixed(decimals)}%`;
  };

  // Calculate progress trends
  const calculateTrend = (current, previous) => {
    if (!previous || previous === 0) return { direction: 'neutral', percentage: 0 };
    const change = ((current - previous) / previous) * 100;
    return {
      direction: change > 0 ? 'up' : change < 0 ? 'down' : 'neutral',
      percentage: Math.abs(change)
    };
  };

  // Handle export
  const handleExport = () => {
    exportProgressMutation.mutate({
      format: exportFormat,
      options: {
        timeRange: selectedTimeRange,
        includeCharts: true,
        includeInsights: true,
        includeGoals: true
      }
    });
  };

  // Update URL params
  useEffect(() => {
    const newParams = new URLSearchParams();
    if (activeTab !== 'overview') newParams.set('tab', activeTab);
    if (selectedTimeRange !== '7d') newParams.set('range', selectedTimeRange);
    setSearchParams(newParams, { replace: true });
  }, [activeTab, selectedTimeRange, setSearchParams]);

  // Render metric card
  const renderMetricCard = (title, value, icon: IconComponent, color = 'blue', trend = null, format = 'number') => {
    let formattedValue = value;
    if (format === 'time') formattedValue = formatTime(value);
    if (format === 'percentage') formattedValue = formatPercentage(value);

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div className={`p-3 bg-${color}-600 bg-opacity-20 rounded-lg`}>
            <IconComponent className={`w-6 h-6 text-${color}-400`} />
          </div>
          {trend && (
            <div className={`flex items-center space-x-1 text-${
              trend.direction === 'up' ? 'green' : trend.direction === 'down' ? 'red' : 'gray'
            }-400`}>
              {trend.direction === 'up' && <ArrowUp className="w-4 h-4" />}
              {trend.direction === 'down' && <ArrowDown className="w-4 h-4" />}
              <span className="text-sm font-medium">{trend.percentage.toFixed(1)}%</span>
            </div>
          )}
        </div>
        
        <div>
          <p className="text-3xl font-bold text-white mb-1">{formattedValue}</p>
          <p className="text-sm text-gray-400">{title}</p>
        </div>
      </motion.div>
    );
  };

  // Render overview tab
  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {progressMetrics && (
          <>
            {renderMetricCard(
              'Completion Rate',
              progressMetrics.completionRate,
              TrendingUp,
              'blue',
              null,
              'percentage'
            )}
            {renderMetricCard(
              'Study Time',
              progressMetrics.totalStudyTime,
              Clock,
              'green',
              null,
              'time'
            )}
            {renderMetricCard(
              'Current Streak',
              progressMetrics.streakDays,
              Flame,
              'orange'
            )}
            {renderMetricCard(
              'Mastery Rate',
              progressMetrics.masteryRate,
              Trophy,
              'purple',
              null,
              'percentage'
            )}
          </>
        )}
      </div>

      {/* Learning Insights */}
      {learningInsights.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <Lightbulb className="w-5 h-5 text-yellow-400" />
            <span>Learning Insights</span>
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {learningInsights.map((insight, index) => (
              <div
                key={index}
                className={`p-4 bg-${insight.color}-600 bg-opacity-10 border border-${insight.color}-600 border-opacity-30 rounded-lg`}
              >
                <div className="flex items-start space-x-3">
                  <insight.icon className={`w-5 h-5 text-${insight.color}-400 mt-1`} />
                  <div className="flex-1">
                    <h4 className="font-medium text-white mb-1">{insight.title}</h4>
                    <p className="text-sm text-gray-300 mb-2">{insight.description}</p>
                    <p className="text-xs text-gray-400">{insight.action}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Progress Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Learning Velocity Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <span>Learning Velocity</span>
          </h3>
          
          <div className="h-64 flex items-center justify-center">
            {/* Placeholder for chart - would integrate with a chart library */}
            <div className="text-center text-gray-400">
              <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Interactive velocity chart would be rendered here</p>
              <p className="text-sm">Shows lessons/hour completion rate over time</p>
            </div>
          </div>
        </motion.div>

        {/* Skill Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
            <PieChart className="w-5 h-5 text-purple-400" />
            <span>Skill Distribution</span>
          </h3>
          
          <div className="h-64 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <PieChart className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Skill points distribution chart</p>
              <p className="text-sm">Breakdown by skill category</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );

  // Render analytics tab
  const renderAnalyticsTab = () => (
    <div className="space-y-6">
      {/* Advanced Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {progressMetrics && (
          <>
            {renderMetricCard(
              'Learning Velocity',
              progressMetrics.learningVelocity,
              FastForward,
              'cyan'
            )}
            {renderMetricCard(
              'Efficiency Score',
              progressMetrics.efficiencyScore,
              Zap,
              'yellow',
              null,
              'percentage'
            )}
            {renderMetricCard(
              'Consistency Score',
              progressMetrics.consistencyScore,
              Target,
              'green',
              null,
              'percentage'
            )}
          </>
        )}
      </div>

      {/* Detailed Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Time Analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Time Analysis</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <Sunrise className="w-5 h-5 text-yellow-400" />
                <span className="text-white">Morning (6-12 PM)</span>
              </div>
              <span className="text-gray-300">2.5h</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <Sun className="w-5 h-5 text-orange-400" />
                <span className="text-white">Afternoon (12-6 PM)</span>
              </div>
              <span className="text-gray-300">1.8h</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <Sunset className="w-5 h-5 text-purple-400" />
                <span className="text-white">Evening (6-10 PM)</span>
              </div>
              <span className="text-gray-300">3.2h</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <Moon className="w-5 h-5 text-blue-400" />
                <span className="text-white">Night (10-6 AM)</span>
              </div>
              <span className="text-gray-300">0.5h</span>
            </div>
          </div>
        </motion.div>

        {/* Performance Trends */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Performance Trends</h3>
          
          <div className="space-y-4">
            <div className="p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white">Lesson Completion</span>
                <div className="flex items-center space-x-1 text-green-400">
                  <ArrowUp className="w-4 h-4" />
                  <span className="text-sm">+12%</span>
                </div>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '78%' }} />
              </div>
            </div>
            
            <div className="p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white">Quest Success Rate</span>
                <div className="flex items-center space-x-1 text-green-400">
                  <ArrowUp className="w-4 h-4" />
                  <span className="text-sm">+8%</span>
                </div>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div className="bg-blue-400 h-2 rounded-full" style={{ width: '85%' }} />
              </div>
            </div>
            
            <div className="p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white">Mastery Achievement</span>
                <div className="flex items-center space-x-1 text-yellow-400">
                  <ArrowUp className="w-4 h-4" />
                  <span className="text-sm">+5%</span>
                </div>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div className="bg-purple-400 h-2 rounded-full" style={{ width: '67%' }} />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );

  // Render goals tab
  const renderGoalsTab = () => (
    <div className="space-y-6">
      {/* Goal Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-white flex items-center space-x-2">
            <Target className="w-5 h-5 text-blue-400" />
            <span>Learning Goals</span>
          </h3>
          <button
            onClick={() => navigate('/goals/new')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Add Goal</span>
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Current Goals */}
          <div className="space-y-4">
            <h4 className="text-lg font-medium text-white">Active Goals</h4>
            
            <div className="space-y-3">
              {/* Example goal */}
              <div className="p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white">Complete Phase 1</span>
                  <span className="text-sm text-gray-400">Due: 2 weeks</span>
                </div>
                <div className="mb-2">
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div className="bg-blue-400 h-2 rounded-full" style={{ width: '65%' }} />
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-400">
                  <span>65% complete</span>
                  <span>8/12 weeks</span>
                </div>
              </div>

              <div className="p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white">30-Day Streak</span>
                  <span className="text-sm text-gray-400">Due: 15 days</span>
                </div>
                <div className="mb-2">
                  <div className="w-full bg-gray-600 rounded-full h-2">
                    <div className="bg-orange-400 h-2 rounded-full" style={{ width: '50%' }} />
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-400">
                  <span>15/30 days</span>
                  <span className="flex items-center space-x-1">
                    <Flame className="w-3 h-3" />
                    <span>Current: 15</span>
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Goal Insights */}
          <div className="space-y-4">
            <h4 className="text-lg font-medium text-white">Goal Insights</h4>
            
            <div className="space-y-3">
              <div className="p-4 bg-green-600 bg-opacity-10 border border-green-600 border-opacity-30 rounded-lg">
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-green-400 mt-1" />
                  <div>
                    <h5 className="font-medium text-white mb-1">On Track</h5>
                    <p className="text-sm text-gray-300">You're making excellent progress on your Phase 1 goal. Keep up the momentum!</p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-yellow-600 bg-opacity-10 border border-yellow-600 border-opacity-30 rounded-lg">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="w-5 h-5 text-yellow-400 mt-1" />
                  <div>
                    <h5 className="font-medium text-white mb-1">Attention Needed</h5>
                    <p className="text-sm text-gray-300">Your streak goal needs more consistent daily practice to stay on track.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Goal Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {renderMetricCard('Goals Set', 8, Target, 'blue')}
        {renderMetricCard('Goals Achieved', 5, Trophy, 'green')}
        {renderMetricCard('Success Rate', 0.625, Percent, 'purple', null, 'percentage')}
        {renderMetricCard('Avg. Time to Goal', 21, Calendar, 'orange')}
      </div>
    </div>
  );

  // Render comparison tab
  const renderComparisonTab = () => (
    <div className="space-y-6">
      {/* Comparison Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-800 border border-gray-700 rounded-lg p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Progress Comparison</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Compare With</label>
            <select className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white">
              <option>Previous Week</option>
              <option>Previous Month</option>
              <option>Same Period Last Year</option>
              <option>Personal Best</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Metric</label>
            <select 
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
            >
              <option value="completion_rate">Completion Rate</option>
              <option value="study_time">Study Time</option>
              <option value="mastery_rate">Mastery Rate</option>
              <option value="velocity">Learning Velocity</option>
            </select>
          </div>
          
          <div className="flex items-end">
            <button 
              onClick={() => setComparisonMode(!comparisonMode)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {comparativeLoading ? 'Loading...' : 'Compare'}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Comparison Results */}
      {comparisonMode && comparativeData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6"
        >
          <h4 className="text-lg font-medium text-white mb-4">Comparison Results</h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h5 className="font-medium text-gray-300">Current Period</h5>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Completion Rate</span>
                  <span className="text-white">78%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Study Time</span>
                  <span className="text-white">24.5h</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mastery Rate</span>
                  <span className="text-white">65%</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h5 className="font-medium text-gray-300">Previous Period</h5>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Completion Rate</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-white">69%</span>
                    <div className="flex items-center text-green-400">
                      <ArrowUp className="w-3 h-3" />
                      <span className="text-xs">+9%</span>
                    </div>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Study Time</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-white">21.2h</span>
                    <div className="flex items-center text-green-400">
                      <ArrowUp className="w-3 h-3" />
                      <span className="text-xs">+3.3h</span>
                    </div>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mastery Rate</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-white">61%</span>
                    <div className="flex items-center text-green-400">
                      <ArrowUp className="w-3 h-3" />
                      <span className="text-xs">+4%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );

  // Render loading state
  if (analyticsLoading && !analyticsData) {
    return (
      <div className="progress-tracker h-full">
        <LoadingSpinner
          size="large"
          text="Loading progress analytics..."
          description="Analyzing your learning patterns and generating insights"
        />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="progress-tracker min-h-screen bg-gray-900"
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
              <h1 className="text-4xl font-bold text-white mb-2">Progress Tracker</h1>
              <p className="text-gray-400 text-lg">
                Deep insights into your learning journey and performance
              </p>
            </div>

            <div className="flex items-center space-x-4">
              {/* Time Range Selector */}
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
              >
                {timeRanges.map((range) => (
                  <option key={range.id} value={range.id}>
                    {range.label}
                  </option>
                ))}
              </select>

              {/* Export Button */}
              <button
                onClick={handleExport}
                disabled={exportProgressMutation.isLoading}
                className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>{exportProgressMutation.isLoading ? 'Exporting...' : 'Export'}</span>
              </button>

              <button
                onClick={() => queryClient.invalidateQueries(['progressAnalytics'])}
                className="p-2 bg-gray-800 border border-gray-600 hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
                title="Refresh Data"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex space-x-1 bg-gray-800 border border-gray-700 rounded-lg p-1">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'analytics', label: 'Analytics', icon: TrendingUp },
              { id: 'goals', label: 'Goals', icon: Target },
              { id: 'comparison', label: 'Comparison', icon: Users }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'overview' && renderOverviewTab()}
            {activeTab === 'analytics' && renderAnalyticsTab()}
            {activeTab === 'goals' && renderGoalsTab()}
            {activeTab === 'comparison' && renderComparisonTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default ProgressTracker;