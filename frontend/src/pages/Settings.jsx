import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Settings as SettingsIcon,
  User,
  Clock,
  Monitor,
  Database,
  Download,
  Upload,
  RotateCcw,
  Save,
  Bell,
  Lock,
  Palette,
  Brain,
  Code,
  Eye,
  Target,
  Zap,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Globe,
  Calendar,
  Award,
  Shield,
  HardDrive,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Info,
  ArrowLeft
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../utils/api';

const Settings = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  // Active section state
  const [activeSection, setActiveSection] = useState('profile');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Settings state
  const [settings, setSettings] = useState({
    // Profile settings
    username: 'Neural Explorer',
    timezone: 'UTC',
    preferred_session_length: 25,
    daily_goal_minutes: 60,
    avatar_style: 'neural_network',
    
    // Appearance settings
    theme: 'dark',
    accent_color: 'blue',
    animations_enabled: true,
    sound_effects: true,
    notifications_enabled: true,
    
    // Learning preferences
    difficulty_adaptation: true,
    hint_system: true,
    auto_save_code: true,
    spaced_repetition: true,
    session_reminders: true,
    
    // Privacy & Data
    analytics_enabled: true,
    backup_frequency: 'daily',
    data_export_format: 'json'
  });

  // Fetch current settings
  const { data: profileData, isLoading } = useQuery(
    'userProfile',
    () => api.get('/auth/profile'), // Assuming we have this endpoint
    {
      onSuccess: (data) => {
        if (data.data) {
          setSettings(prev => ({ ...prev, ...data.data }));
        }
      },
      onError: () => {
        // Use default settings if API fails
        console.log('Using default settings');
      }
    }
  );

  // Fetch database stats
  const { data: dbStats } = useQuery(
    'databaseStats',
    () => api.get('/db/status'),
    { refetchInterval: 30000 }
  );

  // Fetch analytics
  const { data: analyticsData } = useQuery(
    'settingsAnalytics',
    () => api.get('/analytics/summary')
  );

  // Save settings mutation
  const saveSettingsMutation = useMutation(
    (settingsData) => api.put('/auth/profile', settingsData),
    {
      onSuccess: () => {
        setHasUnsavedChanges(false);
        toast.success('Settings saved successfully!');
        queryClient.invalidateQueries('userProfile');
      },
      onError: () => {
        toast.error('Failed to save settings');
      }
    }
  );

  // Export data mutation
  const exportDataMutation = useMutation(
    (format) => api.post('/export/data', { format }),
    {
      onSuccess: (data) => {
        // Create download link
        const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neural-odyssey-export-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Data exported successfully!');
      },
      onError: () => {
        toast.error('Failed to export data');
      }
    }
  );

  // Reset progress mutation
  const resetProgressMutation = useMutation(
    () => api.post('/learning/reset'),
    {
      onSuccess: () => {
        toast.success('Progress reset successfully!');
        queryClient.invalidateQueries();
      },
      onError: () => {
        toast.error('Failed to reset progress');
      }
    }
  );

  // Handle setting change
  const handleSettingChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setHasUnsavedChanges(true);
  };

  // Handle save
  const handleSave = () => {
    saveSettingsMutation.mutate(settings);
  };

  // Handle reset to defaults
  const handleResetToDefaults = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      const defaultSettings = {
        username: 'Neural Explorer',
        timezone: 'UTC',
        preferred_session_length: 25,
        daily_goal_minutes: 60,
        avatar_style: 'neural_network',
        theme: 'dark',
        accent_color: 'blue',
        animations_enabled: true,
        sound_effects: true,
        notifications_enabled: true,
        difficulty_adaptation: true,
        hint_system: true,
        auto_save_code: true,
        spaced_repetition: true,
        session_reminders: true,
        analytics_enabled: true,
        backup_frequency: 'daily',
        data_export_format: 'json'
      };
      setSettings(defaultSettings);
      setHasUnsavedChanges(true);
    }
  };

  // Handle progress reset
  const handleProgressReset = () => {
    if (window.confirm('⚠️ This will permanently delete ALL your learning progress, quest completions, and vault unlocks. This action cannot be undone!\n\nAre you absolutely sure?')) {
      if (window.confirm('Last chance! This will erase your entire Neural Odyssey journey. Type "RESET" to confirm.')) {
        const userInput = window.prompt('Type "RESET" to confirm:');
        if (userInput === 'RESET') {
          resetProgressMutation.mutate();
        }
      }
    }
  };

  // Settings sections
  const settingSections = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'learning', label: 'Learning', icon: Brain },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'data', label: 'Data & Privacy', icon: Database },
    { id: 'advanced', label: 'Advanced', icon: SettingsIcon }
  ];

  // Avatar styles
  const avatarStyles = [
    { id: 'neural_network', label: 'Neural Network', preview: '🧠' },
    { id: 'robot', label: 'Robot', preview: '🤖' },
    { id: 'scientist', label: 'Scientist', preview: '👨‍🔬' },
    { id: 'explorer', label: 'Explorer', preview: '🚀' }
  ];

  // Theme options
  const themes = [
    { id: 'dark', label: 'Dark', preview: '#0a0a0a' },
    { id: 'light', label: 'Light', preview: '#ffffff' },
    { id: 'auto', label: 'Auto', preview: 'linear-gradient(45deg, #0a0a0a, #ffffff)' }
  ];

  // Accent colors
  const accentColors = [
    { id: 'blue', label: 'Blue', color: '#00d4ff' },
    { id: 'green', label: 'Green', color: '#00ff88' },
    { id: 'purple', label: 'Purple', color: '#8855ff' },
    { id: 'orange', label: 'Orange', color: '#ff8800' },
    { id: 'pink', label: 'Pink', color: '#ff44aa' }
  ];

  if (isLoading) {
    return (
      <div className="settings-page h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-blue-400">
          <SettingsIcon className="w-6 h-6 animate-spin" />
          <span>Loading settings...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="settings-page h-full overflow-auto"
    >
      <div className="max-w-6xl mx-auto p-6">
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
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <SettingsIcon className="w-8 h-8 text-blue-400" />
                Settings
              </h1>
              <p className="text-gray-400">Customize your Neural Odyssey experience</p>
            </div>
          </div>

          {/* Save Button */}
          {hasUnsavedChanges && (
            <motion.button
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              onClick={handleSave}
              disabled={saveSettingsMutation.isLoading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Save Changes
            </motion.button>
          )}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1"
          >
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
              <nav className="space-y-2">
                {settingSections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left ${
                      activeSection === section.id
                        ? 'bg-blue-500 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <section.icon className="w-4 h-4" />
                    {section.label}
                  </button>
                ))}
              </nav>
            </div>
          </motion.div>

          {/* Main Content */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-3"
          >
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
              <AnimatePresence mode="wait">
                {/* Profile Settings */}
                {activeSection === 'profile' && (
                  <motion.div
                    key="profile"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <User className="w-6 h-6 text-blue-400" />
                      Profile Settings
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Username */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Username
                        </label>
                        <input
                          type="text"
                          value={settings.username}
                          onChange={(e) => handleSettingChange('username', e.target.value)}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                        />
                      </div>

                      {/* Timezone */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Timezone
                        </label>
                        <select
                          value={settings.timezone}
                          onChange={(e) => handleSettingChange('timezone', e.target.value)}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                        >
                          <option value="UTC">UTC</option>
                          <option value="America/New_York">Eastern Time</option>
                          <option value="America/Chicago">Central Time</option>
                          <option value="America/Denver">Mountain Time</option>
                          <option value="America/Los_Angeles">Pacific Time</option>
                        </select>
                      </div>

                      {/* Session Length */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Preferred Session Length (minutes)
                        </label>
                        <input
                          type="number"
                          min="5"
                          max="120"
                          value={settings.preferred_session_length}
                          onChange={(e) => handleSettingChange('preferred_session_length', parseInt(e.target.value))}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                        />
                      </div>

                      {/* Daily Goal */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Daily Goal (minutes)
                        </label>
                        <input
                          type="number"
                          min="15"
                          max="480"
                          value={settings.daily_goal_minutes}
                          onChange={(e) => handleSettingChange('daily_goal_minutes', parseInt(e.target.value))}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                        />
                      </div>
                    </div>

                    {/* Avatar Style */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Avatar Style
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {avatarStyles.map((style) => (
                          <button
                            key={style.id}
                            onClick={() => handleSettingChange('avatar_style', style.id)}
                            className={`p-4 rounded-lg border-2 transition-colors ${
                              settings.avatar_style === style.id
                                ? 'border-blue-400 bg-blue-400/10'
                                : 'border-gray-600 hover:border-gray-500'
                            }`}
                          >
                            <div className="text-3xl mb-2">{style.preview}</div>
                            <div className="text-sm text-gray-300">{style.label}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Appearance Settings */}
                {activeSection === 'appearance' && (
                  <motion.div
                    key="appearance"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <Palette className="w-6 h-6 text-purple-400" />
                      Appearance Settings
                    </h2>

                    {/* Theme */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Theme
                      </label>
                      <div className="grid grid-cols-3 gap-3">
                        {themes.map((theme) => (
                          <button
                            key={theme.id}
                            onClick={() => handleSettingChange('theme', theme.id)}
                            className={`p-4 rounded-lg border-2 transition-colors ${
                              settings.theme === theme.id
                                ? 'border-blue-400 bg-blue-400/10'
                                : 'border-gray-600 hover:border-gray-500'
                            }`}
                          >
                            <div 
                              className="w-full h-8 rounded mb-2"
                              style={{ background: theme.preview }}
                            />
                            <div className="text-sm text-gray-300">{theme.label}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Accent Color */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Accent Color
                      </label>
                      <div className="grid grid-cols-5 gap-3">
                        {accentColors.map((color) => (
                          <button
                            key={color.id}
                            onClick={() => handleSettingChange('accent_color', color.id)}
                            className={`p-3 rounded-lg border-2 transition-colors ${
                              settings.accent_color === color.id
                                ? 'border-white'
                                : 'border-gray-600 hover:border-gray-500'
                            }`}
                          >
                            <div 
                              className="w-full h-6 rounded mb-1"
                              style={{ backgroundColor: color.color }}
                            />
                            <div className="text-xs text-gray-300">{color.label}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Toggle Options */}
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Animations</div>
                          <div className="text-sm text-gray-400">Enable smooth animations and transitions</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('animations_enabled', !settings.animations_enabled)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.animations_enabled ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.animations_enabled ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Sound Effects</div>
                          <div className="text-sm text-gray-400">Play sounds for interactions and achievements</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('sound_effects', !settings.sound_effects)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.sound_effects ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.sound_effects ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Learning Settings */}
                {activeSection === 'learning' && (
                  <motion.div
                    key="learning"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <Brain className="w-6 h-6 text-green-400" />
                      Learning Preferences
                    </h2>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Adaptive Difficulty</div>
                          <div className="text-sm text-gray-400">Automatically adjust quest difficulty based on performance</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('difficulty_adaptation', !settings.difficulty_adaptation)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.difficulty_adaptation ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.difficulty_adaptation ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Hint System</div>
                          <div className="text-sm text-gray-400">Show helpful hints during quests</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('hint_system', !settings.hint_system)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.hint_system ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.hint_system ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Auto-save Code</div>
                          <div className="text-sm text-gray-400">Automatically save code while you work</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('auto_save_code', !settings.auto_save_code)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.auto_save_code ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.auto_save_code ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Spaced Repetition</div>
                          <div className="text-sm text-gray-400">Enable scientifically-optimized review scheduling</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('spaced_repetition', !settings.spaced_repetition)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.spaced_repetition ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.spaced_repetition ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Data & Privacy */}
                {activeSection === 'data' && (
                  <motion.div
                    key="data"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <Database className="w-6 h-6 text-blue-400" />
                      Data & Privacy
                    </h2>

                    {/* Database Stats */}
                    {dbStats?.data && (
                      <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                        <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                          <HardDrive className="w-4 h-4" />
                          Database Information
                        </h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Lessons Loaded:</span>
                            <span className="text-white ml-2">{dbStats.data.lessons_loaded}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Connection:</span>
                            <span className="text-green-400 ml-2">
                              {dbStats.data.connected ? 'Connected' : 'Disconnected'}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Export Data */}
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                      <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                        <Download className="w-4 h-4" />
                        Export Your Data
                      </h3>
                      <p className="text-gray-400 mb-4 text-sm">
                        Download all your learning progress, quest completions, and settings.
                      </p>
                      <button
                        onClick={() => exportDataMutation.mutate('json')}
                        disabled={exportDataMutation.isLoading}
                        className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Export Data
                      </button>
                    </div>

                    {/* Reset Progress */}
                    <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                      <h3 className="font-semibold text-red-400 mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Danger Zone
                      </h3>
                      <p className="text-gray-400 mb-4 text-sm">
                        Permanently delete all your learning progress. This action cannot be undone.
                      </p>
                      <button
                        onClick={handleProgressReset}
                        disabled={resetProgressMutation.isLoading}
                        className="bg-red-500 hover:bg-red-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset All Progress
                      </button>
                    </div>
                  </motion.div>
                )}

                {/* Advanced Settings */}
                {activeSection === 'advanced' && (
                  <motion.div
                    key="advanced"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <SettingsIcon className="w-6 h-6 text-orange-400" />
                      Advanced Settings
                    </h2>

                    {/* System Information */}
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                      <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                        <Monitor className="w-4 h-4" />
                        System Information
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-400">Browser:</span>
                          <span className="text-white ml-2">{navigator.userAgent.split(' ')[0]}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Platform:</span>
                          <span className="text-white ml-2">{navigator.platform}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Screen Resolution:</span>
                          <span className="text-white ml-2">{screen.width}x{screen.height}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Local Storage:</span>
                          <span className="text-white ml-2">
                            {typeof Storage !== 'undefined' ? 'Available' : 'Not Available'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Cache Management */}
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                      <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                        <RefreshCw className="w-4 h-4" />
                        Cache Management
                      </h3>
                      <p className="text-gray-400 mb-4 text-sm">
                        Clear cached data to free up space or fix loading issues.
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            localStorage.clear();
                            sessionStorage.clear();
                            toast.success('Local cache cleared');
                          }}
                          className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                          <RefreshCw className="w-4 h-4" />
                          Clear Local Cache
                        </button>
                        
                        <button
                          onClick={() => {
                            queryClient.clear();
                            toast.success('Query cache cleared');
                          }}
                          className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                          <RefreshCw className="w-4 h-4" />
                          Clear Query Cache
                        </button>
                      </div>
                    </div>

                    {/* Developer Options */}
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                      <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                        <Code className="w-4 h-4" />
                        Developer Options
                      </h3>
                      
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-white">Debug Mode</div>
                            <div className="text-sm text-gray-400">Show detailed console logs and debug information</div>
                          </div>
                          <button
                            onClick={() => {
                              const debugMode = !localStorage.getItem('debug_mode');
                              localStorage.setItem('debug_mode', debugMode);
                              toast.success(`Debug mode ${debugMode ? 'enabled' : 'disabled'}`);
                            }}
                            className="bg-gray-600 hover:bg-gray-500 text-white px-3 py-1 rounded text-sm transition-colors"
                          >
                            Toggle
                          </button>
                        </div>

                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-white">Performance Monitoring</div>
                            <div className="text-sm text-gray-400">Track render times and performance metrics</div>
                          </div>
                          <button
                            onClick={() => {
                              const perfMode = !localStorage.getItem('perf_mode');
                              localStorage.setItem('perf_mode', perfMode);
                              toast.success(`Performance monitoring ${perfMode ? 'enabled' : 'disabled'}`);
                            }}
                            className="bg-gray-600 hover:bg-gray-500 text-white px-3 py-1 rounded text-sm transition-colors"
                          >
                            Toggle
                          </button>
                        </div>

                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-white">Feature Flags</div>
                            <div className="text-sm text-gray-400">Enable experimental features and beta functionality</div>
                          </div>
                          <button
                            onClick={() => toast.info('Feature flags coming soon!')}
                            className="bg-gray-600 hover:bg-gray-500 text-white px-3 py-1 rounded text-sm transition-colors"
                          >
                            Configure
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Reset Settings */}
                    <div className="bg-orange-900/20 border border-orange-500/30 rounded-lg p-4">
                      <h3 className="font-semibold text-orange-400 mb-3 flex items-center gap-2">
                        <RotateCcw className="w-4 h-4" />
                        Reset Settings
                      </h3>
                      <p className="text-gray-400 mb-4 text-sm">
                        Reset all settings to their default values. Your learning progress will not be affected.
                      </p>
                      <button
                        onClick={handleResetToDefaults}
                        className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset to Defaults
                      </button>
                    </div>
                  </motion.div>
                )}

                {/* Notifications Settings */}
                {activeSection === 'notifications' && (
                  <motion.div
                    key="notifications"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                      <Bell className="w-6 h-6 text-yellow-400" />
                      Notification Settings
                    </h2>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Push Notifications</div>
                          <div className="text-sm text-gray-400">Receive notifications even when the app is closed</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('notifications_enabled', !settings.notifications_enabled)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.notifications_enabled ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.notifications_enabled ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Session Reminders</div>
                          <div className="text-sm text-gray-400">Get reminded to start your daily learning sessions</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('session_reminders', !settings.session_reminders)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.session_reminders ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.session_reminders ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Achievement Notifications</div>
                          <div className="text-sm text-gray-400">Celebrate completed quests and unlocked achievements</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('achievement_notifications', !settings.achievement_notifications)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.achievement_notifications ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.achievement_notifications ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-white">Review Reminders</div>
                          <div className="text-sm text-gray-400">Get notified when spaced repetition reviews are due</div>
                        </div>
                        <button
                          onClick={() => handleSettingChange('review_reminders', !settings.review_reminders)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            settings.review_reminders ? 'bg-blue-500' : 'bg-gray-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              settings.review_reminders ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>

                    {/* Notification Schedule */}
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-600">
                      <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        Notification Schedule
                      </h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Morning Reminder
                          </label>
                          <input
                            type="time"
                            value="09:00"
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Evening Reminder
                          </label>
                          <input
                            type="time"
                            value="19:00"
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-blue-400 focus:outline-none"
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

export default Settings;