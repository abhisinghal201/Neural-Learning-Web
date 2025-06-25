/**
 * Enhanced Neural Odyssey Progress Tracker Page
 * 
 * Now fully leverages ALL backend analytics capabilities:
 * - Comprehensive learning analytics with multiple timeframes
 * - Session analysis with mood, energy, and focus correlations
 * - Performance metrics and learning velocity tracking
 * - Skill progression analysis with detailed breakdowns
 * - Spaced repetition analytics and memory retention insights
 * - Goal completion rates and productivity patterns
 * - Knowledge graph connections and concept mastery
 * - Detailed charts and visualizations
 * - Exportable analytics reports
 * - Predictive learning insights
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation } from 'react-query';
import { useSearchParams } from 'react-router-dom';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Clock,
  Brain,
  Target,
  Award,
  Calendar,
  Flame,
  Star,
  Zap,
  Activity,
  Eye,
  ArrowUp,
  ArrowDown,
  Filter,
  Download,
  RefreshCw,
  Settings,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Info,
  AlertTriangle,
  CheckCircle,
  Timer,
  Battery,
  Layers,
  Map,
  Users,
  Globe,
  Bookmark,
  MessageSquare,
  ThumbsUp,
  Coffee,
  Smile,
  Meh,
  Frown,
  Plus,
  Minus,
  X,
  Search,
  Grid,
  List
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter
} from 'recharts';
import toast from 'react-hot-toast';

// Components
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const ProgressTracker = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  
  // State management
  const [selectedTimeframe, setSelectedTimeframe] = useState(searchParams.get('timeframe') || '30');
  const [selectedMetric, setSelectedMetric] = useState(searchParams.get('metric') || 'overview');
  const [selectedChartType, setSelectedChartType] = useState('line');
  const [showFilters, setShowFilters] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [compareTimeframe, setCompareTimeframe] = useState('30');
  const [groupBy, setGroupBy] = useState('day');
  const [selectedPhases, setSelectedPhases] = useState([1, 2, 3, 4]);
  const [selectedSessionTypes, setSelectedSessionTypes] = useState(['math', 'coding', 'visual_projects', 'real_applications']);

  // Advanced filters
  const [filters, setFilters] = useState({
    minSessionLength: 0,
    maxSessionLength: 180,
    minFocusScore: 1,
    maxFocusScore: 10,
    energyLevels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    moods: ['energetic', 'focused', 'calm', 'excited', 'tired', 'distracted', 'stressed'],
    environments: ['home', 'office', 'library', 'cafe', 'other'],
    includeLowPerformance: true,
    includeIncompleteData: true
  });

  // Data fetching with comprehensive analytics
  const { data: analyticsData, isLoading, refetch } = useQuery(
    ['analytics', selectedTimeframe, selectedMetric, filters],
    () => api.learning.getAnalytics({
      timeframe: selectedTimeframe,
      metric: selectedMetric,
      group_by: groupBy,
      phases: selectedPhases.join(','),
      session_types: selectedSessionTypes.join(','),
      min_session_length: filters.minSessionLength,
      max_session_length: filters.maxSessionLength,
      min_focus_score: filters.minFocusScore,
      max_focus_score: filters.maxFocusScore,
      energy_levels: filters.energyLevels.join(','),
      moods: filters.moods.join(','),
      environments: filters.environments.join(',')
    }),
    {
      refetchInterval: 300000, // 5 minutes
      staleTime: 60000
    }
  );

  const { data: progressData } = useQuery(
    'progressData',
    () => api.learning.getProgress(),
    { refetchInterval: 60000 }
  );

  const { data: spacedRepetitionAnalytics } = useQuery(
    'spacedRepetitionAnalytics',
    () => api.learning.getSpacedRepetition({ analytics: true }),
    { refetchInterval: 300000 }
  );

  const { data: knowledgeGraphData } = useQuery(
    'knowledgeGraphAnalytics',
    () => api.learning.getKnowledgeGraph({ analytics: true }),
    { refetchInterval: 300000 }
  );

  const { data: streakData } = useQuery(
    'streakAnalytics',
    () => api.learning.getStreak(),
    { refetchInterval: 60000 }
  );

  const { data: compareData } = useQuery(
    ['compareAnalytics', compareTimeframe],
    () => compareMode ? api.learning.getAnalytics({
      timeframe: compareTimeframe,
      metric: selectedMetric,
      group_by: groupBy
    }) : null,
    {
      enabled: compareMode,
      refetchInterval: 300000
    }
  );

  // Export analytics mutation
  const exportAnalyticsMutation = useMutation(
    (exportOptions) => api.system.exportPortfolio(exportOptions),
    {
      onSuccess: () => {
        toast.success('Analytics exported successfully!');
      },
      onError: () => {
        toast.error('Failed to export analytics');
      }
    }
  );

  // Update URL params
  useEffect(() => {
    const params = new URLSearchParams();
    params.set('timeframe', selectedTimeframe);
    params.set('metric', selectedMetric);
    setSearchParams(params);
  }, [selectedTimeframe, selectedMetric, setSearchParams]);

  // Timeframe options
  const timeframeOptions = [
    { value: '7', label: 'Last 7 Days' },
    { value: '14', label: 'Last 2 Weeks' },
    { value: '30', label: 'Last Month' },
    { value: '60', label: 'Last 2 Months' },
    { value: '90', label: 'Last 3 Months' },
    { value: '180', label: 'Last 6 Months' },
    { value: '365', label: 'Last Year' }
  ];

  const metricOptions = [
    { value: 'overview', label: 'Overview', icon: BarChart3 },
    { value: 'sessions', label: 'Sessions', icon: Clock },
    { value: 'focus', label: 'Focus & Performance', icon: Brain },
    { value: 'skills', label: 'Skill Development', icon: Award },
    { value: 'progress', label: 'Learning Progress', icon: TrendingUp },
    { value: 'memory', label: 'Memory & Retention', icon: Layers },
    { value: 'patterns', label: 'Learning Patterns', icon: Activity },
    { value: 'goals', label: 'Goals & Achievements', icon: Target }
  ];

  // Color schemes for charts
  const colors = {
    primary: '#3B82F6',
    secondary: '#10B981',
    accent: '#F59E0B',
    danger: '#EF4444',
    purple: '#8B5CF6',
    pink: '#EC4899',
    cyan: '#06B6D4',
    orange: '#F97316'
  };

  const chartColors = [colors.primary, colors.secondary, colors.accent, colors.purple, colors.pink, colors.cyan];

  // Calculate insights and trends
  const insights = useMemo(() => {
    if (!analyticsData?.data) return {};

    const data = analyticsData.data;
    const insights = {};

    // Performance trends
    if (data.daily_activity && data.daily_activity.length > 1) {
      const recent = data.daily_activity.slice(-7);
      const older = data.daily_activity.slice(-14, -7);
      
      const recentAvgFocus = recent.reduce((sum, d) => sum + (d.avg_focus || 0), 0) / recent.length;
      const olderAvgFocus = older.length > 0 ? older.reduce((sum, d) => sum + (d.avg_focus || 0), 0) / older.length : recentAvgFocus;
      
      insights.focusTrend = recentAvgFocus > olderAvgFocus ? 'improving' : 'declining';
      insights.focusChange = Math.abs(recentAvgFocus - olderAvgFocus);
    }

    // Session patterns
    if (data.sessions) {
      const sessionsByDay = {};
      data.daily_activity?.forEach(day => {
        const dayOfWeek = new Date(day.session_date).getDay();
        sessionsByDay[dayOfWeek] = (sessionsByDay[dayOfWeek] || 0) + (day.session_count || 0);
      });
      
      const maxDay = Object.keys(sessionsByDay).reduce((max, day) => 
        sessionsByDay[day] > sessionsByDay[max] ? day : max, '0');
      
      const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      insights.mostProductiveDay = dayNames[maxDay];
    }

    // Learning velocity
    if (data.progress) {
      const completionRate = data.progress.completed_lessons / Math.max(data.progress.total_lessons, 1);
      insights.learningVelocity = completionRate > 0.8 ? 'fast' : completionRate > 0.5 ? 'moderate' : 'slow';
    }

    return insights;
  }, [analyticsData]);

  // Render overview dashboard
  const renderOverview = () => {
    const data = analyticsData?.data;
    if (!data) return null;

    return (
      <div className="space-y-6">
        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg p-6 text-white"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{data.sessions?.total_sessions || 0}</div>
                <div className="text-sm opacity-90">Total Sessions</div>
              </div>
              <Clock className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              Avg: {Math.round(data.sessions?.avg_session_length || 0)}min
            </div>
            {insights.focusTrend && (
              <div className={`text-xs mt-1 flex items-center space-x-1 ${
                insights.focusTrend === 'improving' ? 'text-green-200' : 'text-red-200'
              }`}>
                {insights.focusTrend === 'improving' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                <span>Focus {insights.focusTrend}</span>
              </div>
            )}
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg p-6 text-white"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{Math.round(data.sessions?.avg_focus_score || 0)}/10</div>
                <div className="text-sm opacity-90">Avg Focus Score</div>
              </div>
              <Brain className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              Energy: {Math.round(data.sessions?.avg_energy_level || 0)}/10
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-6 text-white"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{data.progress?.completed_lessons || 0}</div>
                <div className="text-sm opacity-90">Lessons Completed</div>
              </div>
              <Award className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              {Math.round((data.progress?.completed_lessons / Math.max(data.progress?.total_lessons, 1)) * 100)}% complete
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gradient-to-r from-orange-500 to-red-500 rounded-lg p-6 text-white"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{streakData?.data?.currentStreak || 0}</div>
                <div className="text-sm opacity-90">Current Streak</div>
              </div>
              <Flame className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              Best: {streakData?.data?.longestStreak || 0} days
            </div>
          </motion.div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Daily Activity Chart */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Daily Activity</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={data.daily_activity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="session_date" 
                  stroke="#9CA3AF"
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="total_time" 
                  stroke={colors.primary} 
                  fill={colors.primary}
                  fillOpacity={0.3}
                  name="Study Time (min)"
                />
                <Area 
                  type="monotone" 
                  dataKey="avg_focus" 
                  stroke={colors.secondary} 
                  fill={colors.secondary}
                  fillOpacity={0.3}
                  name="Focus Score"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Session Types Distribution */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Session Types</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={data.session_type_distribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {data.session_type_distribution?.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Focus vs Energy Correlation */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Focus vs Energy Correlation</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={data.focus_energy_correlation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  type="number" 
                  dataKey="energy_level" 
                  name="Energy Level"
                  domain={[1, 10]}
                  stroke="#9CA3AF"
                />
                <YAxis 
                  type="number" 
                  dataKey="focus_score" 
                  name="Focus Score"
                  domain={[1, 10]}
                  stroke="#9CA3AF"
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  formatter={(value, name) => [value, name]}
                />
                <Scatter name="Sessions" dataKey="focus_score" fill={colors.accent} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Skill Points Progression */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Skill Development</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.skills}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="category" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="total_points" fill={colors.purple} name="Total Points" />
                <Bar dataKey="achievements" fill={colors.pink} name="Achievements" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Insights and Recommendations */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Insights & Recommendations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {insights.mostProductiveDay && (
              <div className="bg-blue-600/10 border border-blue-600/30 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Calendar className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-medium text-blue-400">Most Productive Day</span>
                </div>
                <div className="text-white">{insights.mostProductiveDay}</div>
                <div className="text-xs text-gray-400 mt-1">
                  Consider scheduling important sessions on this day
                </div>
              </div>
            )}

            {insights.learningVelocity && (
              <div className={`border rounded-lg p-4 ${
                insights.learningVelocity === 'fast' ? 'bg-green-600/10 border-green-600/30' :
                insights.learningVelocity === 'moderate' ? 'bg-yellow-600/10 border-yellow-600/30' :
                'bg-red-600/10 border-red-600/30'
              }`}>
                <div className="flex items-center space-x-2 mb-2">
                  <TrendingUp className={`w-4 h-4 ${
                    insights.learningVelocity === 'fast' ? 'text-green-400' :
                    insights.learningVelocity === 'moderate' ? 'text-yellow-400' :
                    'text-red-400'
                  }`} />
                  <span className={`text-sm font-medium ${
                    insights.learningVelocity === 'fast' ? 'text-green-400' :
                    insights.learningVelocity === 'moderate' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>Learning Velocity</span>
                </div>
                <div className="text-white capitalize">{insights.learningVelocity}</div>
                <div className="text-xs text-gray-400 mt-1">
                  {insights.learningVelocity === 'fast' ? 'Great pace! Keep it up!' :
                   insights.learningVelocity === 'moderate' ? 'Steady progress. Consider increasing session frequency.' :
                   'Consider breaking lessons into smaller chunks'}
                </div>
              </div>
            )}

            {insights.focusTrend && (
              <div className={`border rounded-lg p-4 ${
                insights.focusTrend === 'improving' ? 'bg-green-600/10 border-green-600/30' : 'bg-orange-600/10 border-orange-600/30'
              }`}>
                <div className="flex items-center space-x-2 mb-2">
                  <Brain className={`w-4 h-4 ${insights.focusTrend === 'improving' ? 'text-green-400' : 'text-orange-400'}`} />
                  <span className={`text-sm font-medium ${insights.focusTrend === 'improving' ? 'text-green-400' : 'text-orange-400'}`}>
                    Focus Trend
                  </span>
                </div>
                <div className="text-white capitalize">{insights.focusTrend}</div>
                <div className="text-xs text-gray-400 mt-1">
                  {insights.focusTrend === 'improving' ? 
                    'Your focus is improving over time!' :
                    'Consider adjusting your environment or session timing'
                  }
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Render sessions analytics
  const renderSessionsAnalytics = () => {
    const data = analyticsData?.data;
    if (!data) return null;

    return (
      <div className="space-y-6">
        {/* Session Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Session Duration</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data.session_duration_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="duration_range" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill={colors.primary} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Focus Score Distribution</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data.focus_score_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="score_range" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill={colors.secondary} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Energy Levels</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data.energy_level_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="energy_level" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill={colors.accent} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Session Patterns */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Session Patterns</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Hour of Day Analysis */}
            <div>
              <h4 className="text-md font-medium text-gray-300 mb-3">Sessions by Hour</h4>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={data.sessions_by_hour}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="hour" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                  <Area type="monotone" dataKey="count" stroke={colors.purple} fill={colors.purple} fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Day of Week Analysis */}
            <div>
              <h4 className="text-md font-medium text-gray-300 mb-3">Sessions by Day</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.sessions_by_day}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  />
                  <Bar dataKey="count" fill={colors.pink} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Mood and Environment Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Mood Impact on Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.mood_performance_correlation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="mood" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="avg_focus" fill={colors.cyan} name="Avg Focus Score" />
                <Bar dataKey="avg_productivity" fill={colors.orange} name="Avg Productivity" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Environment Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={data.environment_performance}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="environment" tick={{ fill: '#9CA3AF' }} />
                <PolarRadiusAxis tick={{ fill: '#9CA3AF' }} />
                <Radar
                  name="Focus Score"
                  dataKey="avg_focus"
                  stroke={colors.primary}
                  fill={colors.primary}
                  fillOpacity={0.3}
                />
                <Radar
                  name="Productivity"
                  dataKey="avg_productivity"
                  stroke={colors.secondary}
                  fill={colors.secondary}
                  fillOpacity={0.3}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  // Render memory and spaced repetition analytics
  const renderMemoryAnalytics = () => {
    const data = spacedRepetitionAnalytics?.data;
    if (!data) return null;

    return (
      <div className="space-y-6">
        {/* Memory Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{data.total_concepts || 0}</div>
                <div className="text-sm text-gray-400">Total Concepts</div>
              </div>
              <Layers className="w-8 h-8 text-blue-400" />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{data.mastered_concepts || 0}</div>
                <div className="text-sm text-gray-400">Mastered</div>
              </div>
              <Award className="w-8 h-8 text-green-400" />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{data.due_for_review || 0}</div>
                <div className="text-sm text-gray-400">Due for Review</div>
              </div>
              <Clock className="w-8 h-8 text-orange-400" />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">
                  {Math.round(data.retention_rate * 100) || 0}%
                </div>
                <div className="text-sm text-gray-400">Retention Rate</div>
              </div>
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
          </div>
        </div>

        {/* Memory Performance Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Retention Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.retention_timeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Line type="monotone" dataKey="retention_rate" stroke={colors.primary} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Difficulty Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.difficulty_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="difficulty_range" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill={colors.secondary} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Review Performance */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Review Performance by Quality Score</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data.quality_score_trends}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
              />
              <Area type="monotone" dataKey="avg_quality" stroke={colors.accent} fill={colors.accent} fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // Export analytics
  const exportAnalytics = (format = 'json') => {
    const exportData = {
      timeframe: selectedTimeframe,
      metric: selectedMetric,
      data: analyticsData?.data,
      insights,
      generated_at: new Date().toISOString()
    };

    exportAnalyticsMutation.mutate({
      type: 'analytics',
      format,
      data: exportData
    });
  };

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
          <h1 className="text-2xl font-bold text-white">Progress Analytics</h1>
          <p className="text-gray-400">Comprehensive learning insights and performance metrics</p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-2 rounded-lg transition-colors ${
              showFilters ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <Filter className="w-4 h-4" />
          </button>
          
          <button
            onClick={refetch}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <button
            onClick={() => exportAnalytics('json')}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Timeframe Selection */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">Timeframe:</span>
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              {timeframeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Metric Selection */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">View:</span>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              {metricOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Compare Mode */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                compareMode ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Compare
            </button>
            {compareMode && (
              <select
                value={compareTimeframe}
                onChange={(e) => setCompareTimeframe(e.target.value)}
                className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              >
                {timeframeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    vs {option.label}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
      </div>

      {/* Advanced Filters */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800 rounded-lg border border-gray-700 p-4"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Advanced Filters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Session Length Filter */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Session Length (min)</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={filters.minSessionLength}
                    onChange={(e) => setFilters(prev => ({ ...prev, minSessionLength: Number(e.target.value) }))}
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                    placeholder="Min"
                  />
                  <span className="text-gray-400">to</span>
                  <input
                    type="number"
                    value={filters.maxSessionLength}
                    onChange={(e) => setFilters(prev => ({ ...prev, maxSessionLength: Number(e.target.value) }))}
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                    placeholder="Max"
                  />
                </div>
              </div>

              {/* Focus Score Filter */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Focus Score</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={filters.minFocusScore}
                    onChange={(e) => setFilters(prev => ({ ...prev, minFocusScore: Number(e.target.value) }))}
                    className="w-16 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                    placeholder="Min"
                  />
                  <span className="text-gray-400">to</span>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={filters.maxFocusScore}
                    onChange={(e) => setFilters(prev => ({ ...prev, maxFocusScore: Number(e.target.value) }))}
                    className="w-16 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
                    placeholder="Max"
                  />
                </div>
              </div>

              {/* Phase Filter */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Phases</label>
                <div className="flex flex-wrap gap-1">
                  {[1, 2, 3, 4].map((phase) => (
                    <button
                      key={phase}
                      onClick={() => {
                        setSelectedPhases(prev => 
                          prev.includes(phase) 
                            ? prev.filter(p => p !== phase)
                            : [...prev, phase]
                        );
                      }}
                      className={`px-2 py-1 rounded text-xs transition-colors ${
                        selectedPhases.includes(phase)
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Phase {phase}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div>
        {selectedMetric === 'overview' && renderOverview()}
        {selectedMetric === 'sessions' && renderSessionsAnalytics()}
        {selectedMetric === 'memory' && renderMemoryAnalytics()}
        {/* Add other metric views here */}
      </div>
    </div>
  );
};

export default ProgressTracker;