/**
 * Neural Odyssey Quest Detail Page
 *
 * Comprehensive quest detail interface for in-depth quest management,
 * code submission, testing, and progress tracking. Provides immersive
 * quest completion experience with real-time feedback and guidance.
 *
 * Features:
 * - Detailed quest information and requirements
 * - Integrated code playground with testing
 * - Real-time progress tracking and hints
 * - Step-by-step guidance and tutorials
 * - Code validation and automated testing
 * - AI mentor feedback and suggestions
 * - Achievement and skill point rewards
 * - Social features and code sharing
 * - Comprehensive analytics and insights
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  Circle,
  Clock,
  Target,
  Brain,
  Code,
  Eye,
  BookOpen,
  Star,
  Award,
  Trophy,
  Lightbulb,
  Zap,
  Flame,
  Users,
  Share2,
  Download,
  Upload,
  Save,
  Settings,
  HelpCircle,
  AlertCircle,
  Info,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  MoreHorizontal,
  RefreshCw,
  Timer,
  Activity,
  BarChart3,
  TrendingUp,
  Calendar,
  Hash,
  Tag,
  Bookmark,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  ExternalLink,
  FileText,
  Clipboard,
  Check,
  X,
  Plus,
  Minus,
  GitBranch,
  Database,
  Terminal,
  Cpu,
  Shield,
  Lock,
  Unlock
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import CodePlayground from '../components/CodePlayground';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import VaultRevealModal from '../components/VaultRevealModal';

// Utils
import { api } from '../utils/api';

const QuestDetail = () => {
  const { questId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // Refs for scroll management
  const contentRef = useRef(null);
  const playgroundRef = useRef(null);

  // State management
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'overview');
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [userCode, setUserCode] = useState('');
  const [testResults, setTestResults] = useState(null);
  const [showHints, setShowHints] = useState(false);
  const [hintsUsed, setHintsUsed] = useState(0);
  const [showSolution, setShowSolution] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState(Date.now());
  const [timeSpent, setTimeSpent] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [expandedSections, setExpandedSections] = useState(new Set(['description']));
  const [selectedDifficulty, setSelectedDifficulty] = useState('standard');
  const [vaultUnlock, setVaultUnlock] = useState(null);
  const [showNotes, setShowNotes] = useState(false);
  const [userNotes, setUserNotes] = useState('');

  // Data fetching
  const { data: questData, isLoading: questLoading, error: questError } = useQuery(
    ['quest', questId],
    () => api.get(`/learning/quests/${questId}`),
    {
      retry: 3,
      refetchInterval: false,
      onSuccess: (data) => {
        if (data.data.userCode) {
          setUserCode(data.data.userCode);
        }
        if (data.data.hintsUsed) {
          setHintsUsed(data.data.hintsUsed);
        }
        if (data.data.timeSpent) {
          setTimeSpent(data.data.timeSpent);
        }
        if (data.data.attempts) {
          setAttempts(data.data.attempts);
        }
        if (data.data.userNotes) {
          setUserNotes(data.data.userNotes);
        }
      }
    }
  );

  const { data: progressData } = useQuery(
    ['questProgress', questId],
    () => api.get(`/learning/quests/${questId}/progress`),
    {
      refetchInterval: 30000
    }
  );

  const { data: hintsData } = useQuery(
    ['questHints', questId],
    () => api.get(`/learning/quests/${questId}/hints`),
    {
      enabled: showHints
    }
  );

  const { data: relatedQuests } = useQuery(
    ['relatedQuests', questId],
    () => api.get(`/learning/quests/${questId}/related`),
    {
      refetchInterval: false
    }
  );

  // Submit quest solution
  const submitQuestMutation = useMutation(
    (submissionData) => api.post(`/learning/quests/${questId}/submit`, submissionData),
    {
      onMutate: () => {
        setIsSubmitting(true);
        setAttempts(prev => prev + 1);
      },
      onSuccess: (data) => {
        setTestResults(data.data.testResults);
        queryClient.invalidateQueries(['quest', questId]);
        queryClient.invalidateQueries(['questProgress', questId]);
        queryClient.invalidateQueries(['learningProgress']);
        
        if (data.data.completed) {
          toast.success('🎉 Quest completed successfully!');
          
          // Check for vault unlocks
          if (data.data.vaultUnlocks && data.data.vaultUnlocks.length > 0) {
            setVaultUnlock(data.data.vaultUnlocks[0]);
          }
        } else if (data.data.testResults) {
          const passedTests = data.data.testResults.filter(t => t.passed).length;
          const totalTests = data.data.testResults.length;
          toast.info(`${passedTests}/${totalTests} tests passed. Keep trying!`);
        }
      },
      onError: (error) => {
        toast.error('Failed to submit quest. Please try again.');
        console.error('Quest submission error:', error);
      },
      onSettled: () => {
        setIsSubmitting(false);
      }
    }
  );

  // Save progress mutation
  const saveProgressMutation = useMutation(
    (progressData) => api.put(`/learning/quests/${questId}/progress`, progressData),
    {
      onSuccess: () => {
        toast.success('Progress saved');
      },
      onError: () => {
        toast.error('Failed to save progress');
      }
    }
  );

  // Use hint mutation
  const useHintMutation = useMutation(
    () => api.post(`/learning/quests/${questId}/hint`),
    {
      onSuccess: () => {
        setHintsUsed(prev => prev + 1);
        queryClient.invalidateQueries(['questHints', questId]);
      }
    }
  );

  // Timer effect
  useEffect(() => {
    const interval = setInterval(() => {
      setTimeSpent(Date.now() - sessionStartTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [sessionStartTime]);

  // Auto-save code
  useEffect(() => {
    const autoSaveTimeout = setTimeout(() => {
      if (userCode && quest) {
        saveProgressMutation.mutate({
          userCode,
          timeSpent,
          hintsUsed,
          currentStep,
          userNotes
        });
      }
    }, 2000);

    return () => clearTimeout(autoSaveTimeout);
  }, [userCode, userNotes]);

  // URL params sync
  useEffect(() => {
    const params = new URLSearchParams(searchParams);
    params.set('tab', activeTab);
    setSearchParams(params, { replace: true });
  }, [activeTab]);

  // Quest data
  const quest = questData?.data;
  const progress = progressData?.data;

  // Quest configuration
  const questTypes = {
    coding_exercise: {
      name: 'Coding Exercise',
      icon: Code,
      color: 'from-blue-500 to-blue-600',
      description: 'Hands-on programming challenge'
    },
    implementation_project: {
      name: 'Implementation Project',
      icon: Brain,
      color: 'from-green-500 to-green-600',
      description: 'Complete implementation task'
    },
    theory_quiz: {
      name: 'Theory Quiz',
      icon: BookOpen,
      color: 'from-purple-500 to-purple-600',
      description: 'Conceptual understanding assessment'
    },
    practical_application: {
      name: 'Practical Application',
      icon: Target,
      color: 'from-orange-500 to-orange-600',
      description: 'Real-world application challenge'
    }
  };

  // Difficulty levels
  const difficultyLevels = {
    beginner: { name: 'Beginner', color: 'text-green-400', stars: 1 },
    intermediate: { name: 'Intermediate', color: 'text-yellow-400', stars: 3 },
    advanced: { name: 'Advanced', color: 'text-red-400', stars: 5 },
    expert: { name: 'Expert', color: 'text-purple-400', stars: 5 }
  };

  // Tab configuration
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'instructions', label: 'Instructions', icon: FileText },
    { id: 'code', label: 'Code', icon: Code },
    { id: 'tests', label: 'Tests', icon: CheckCircle },
    { id: 'hints', label: 'Hints', icon: Lightbulb },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 }
  ];

  // Handle code submission
  const handleSubmit = () => {
    if (!userCode.trim()) {
      toast.error('Please write some code before submitting');
      return;
    }

    submitQuestMutation.mutate({
      code: userCode,
      language: 'python',
      timeSpent,
      hintsUsed,
      difficulty: selectedDifficulty,
      userNotes
    });
  };

  // Handle hint request
  const handleHintRequest = () => {
    if (hintsUsed >= (quest?.maxHints || 3)) {
      toast.error('No more hints available for this quest');
      return;
    }

    useHintMutation.mutate();
    setShowHints(true);
  };

  // Format time display
  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  // Toggle section expansion
  const toggleSection = (sectionId) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  // Render quest header
  const renderQuestHeader = () => {
    if (!quest) return null;

    const typeConfig = questTypes[quest.questType] || questTypes.coding_exercise;
    const difficultyConfig = difficultyLevels[quest.difficulty] || difficultyLevels.intermediate;
    const TypeIcon = typeConfig.icon;

    return (
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700"
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate('/quests')}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            
            <div className={`p-3 rounded-lg bg-gradient-to-br ${typeConfig.color}`}>
              <TypeIcon className="w-6 h-6 text-white" />
            </div>
            
            <div>
              <h1 className="text-2xl font-bold text-white">{quest.title}</h1>
              <div className="flex items-center space-x-4 mt-1">
                <span className="text-sm text-gray-400">{typeConfig.name}</span>
                <div className="flex items-center space-x-1">
                  {Array.from({ length: 5 }, (_, i) => (
                    <Star
                      key={i}
                      className={`w-3 h-3 ${
                        i < difficultyConfig.stars
                          ? `${difficultyConfig.color} fill-current`
                          : 'text-gray-600'
                      }`}
                    />
                  ))}
                  <span className={`text-xs ml-1 ${difficultyConfig.color}`}>
                    {difficultyConfig.name}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Status Badge */}
            <div className={`px-3 py-1 rounded-full text-xs font-medium ${
              quest.status === 'completed' 
                ? 'bg-green-600 text-white'
                : quest.status === 'in_progress'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-600 text-gray-300'
            }`}>
              {quest.status === 'completed' ? 'Completed' : 
               quest.status === 'in_progress' ? 'In Progress' : 'Not Started'}
            </div>

            {/* Actions */}
            <button
              onClick={() => setShowNotes(!showNotes)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <MessageSquare className="w-5 h-5" />
            </button>
            
            <button className="text-gray-400 hover:text-white transition-colors">
              <Share2 className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Quest Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{formatTime(timeSpent)}</div>
            <div className="text-xs text-gray-400">Time Spent</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{attempts}</div>
            <div className="text-xs text-gray-400">Attempts</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{hintsUsed}/{quest.maxHints || 3}</div>
            <div className="text-xs text-gray-400">Hints Used</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{quest.pointReward || 0}</div>
            <div className="text-xs text-gray-400">Points Reward</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{quest.estimatedTime || 'N/A'}</div>
            <div className="text-xs text-gray-400">Est. Time</div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Render tab navigation
  const renderTabNavigation = () => {
    return (
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="bg-gray-800 rounded-lg p-1 mb-6 border border-gray-700"
      >
        <div className="flex space-x-1">
          {tabs.map((tab) => {
            const TabIcon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors flex-1
                  ${isActive 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }
                `}
              >
                <TabIcon className="w-4 h-4" />
                <span>{tab.label}</span>
                {tab.id === 'hints' && hintsUsed > 0 && (
                  <span className="bg-yellow-500 text-black text-xs px-1.5 py-0.5 rounded-full">
                    {hintsUsed}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </motion.div>
    );
  };

  // Render overview tab
  const renderOverviewTab = () => {
    if (!quest) return null;

    return (
      <div className="space-y-6">
        {/* Description */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-3">Quest Description</h3>
          <p className="text-gray-300 leading-relaxed">{quest.description}</p>
        </div>

        {/* Learning Objectives */}
        {quest.learningObjectives && quest.learningObjectives.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-400" />
              Learning Objectives
            </h3>
            <ul className="space-y-2">
              {quest.learningObjectives.map((objective, index) => (
                <li key={index} className="flex items-start space-x-2 text-gray-300">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Prerequisites */}
        {quest.prerequisites && quest.prerequisites.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-400" />
              Prerequisites
            </h3>
            <ul className="space-y-2">
              {quest.prerequisites.map((prereq, index) => (
                <li key={index} className="flex items-start space-x-2 text-gray-300">
                  <Info className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                  <span>{prereq}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Tags */}
        {quest.tags && quest.tags.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Tag className="w-5 h-5 text-purple-400" />
              Tags
            </h3>
            <div className="flex flex-wrap gap-2">
              {quest.tags.map((tag, index) => (
                <span
                  key={index}
                  className="bg-gray-700 text-gray-300 px-3 py-1 rounded-full text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render instructions tab
  const renderInstructionsTab = () => {
    if (!quest) return null;

    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="prose prose-invert max-w-none">
          <div 
            className="text-gray-300 leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: quest.instructions?.replace(/\n/g, '<br>') || 'No detailed instructions available.'
            }}
          />
        </div>
      </div>
    );
  };

  // Render code tab
  const renderCodeTab = () => {
    return (
      <div className="space-y-6">
        {/* Code Playground */}
        <div ref={playgroundRef}>
          <CodePlayground
            initialCode={userCode}
            language="python"
            onCodeChange={setUserCode}
            testCases={quest?.testCases}
            className="h-96"
            showLineNumbers={true}
            showMinimap={true}
            onTestResults={setTestResults}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !userCode.trim()}
              className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              {isSubmitting ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isSubmitting ? 'Submitting...' : 'Submit Solution'}
            </button>

            <button
              onClick={() => setUserCode('')}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleHintRequest}
              disabled={hintsUsed >= (quest?.maxHints || 3)}
              className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Lightbulb className="w-4 h-4" />
              Hint ({hintsUsed}/{quest?.maxHints || 3})
            </button>

            <button
              onClick={() => setShowSolution(!showSolution)}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Eye className="w-4 h-4" />
              {showSolution ? 'Hide' : 'Show'} Solution
            </button>
          </div>
        </div>

        {/* Test Results */}
        {testResults && (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              Test Results
            </h3>
            <div className="space-y-3">
              {testResults.map((result, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border ${
                    result.passed 
                      ? 'bg-green-900 border-green-600' 
                      : 'bg-red-900 border-red-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-white">
                      Test {index + 1}: {result.description || `Test Case ${index + 1}`}
                    </span>
                    <span className={`text-sm font-medium ${
                      result.passed ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {result.passed ? 'PASSED' : 'FAILED'}
                    </span>
                  </div>
                  {result.output && (
                    <div className="text-sm text-gray-300 font-mono bg-gray-800 p-2 rounded">
                      Output: {result.output}
                    </div>
                  )}
                  {result.expected && !result.passed && (
                    <div className="text-sm text-gray-300 font-mono bg-gray-800 p-2 rounded mt-1">
                      Expected: {result.expected}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Solution Display */}
        <AnimatePresence>
          {showSolution && quest?.solutionCode && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-gray-800 rounded-lg p-6 border border-gray-700"
            >
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-purple-400" />
                Reference Solution
              </h3>
              <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
                <pre className="text-gray-300 font-mono text-sm">
                  <code>{quest.solutionCode}</code>
                </pre>
              </div>
              {quest.solutionExplanation && (
                <div className="mt-4 p-4 bg-purple-900 border border-purple-600 rounded-lg">
                  <h4 className="font-medium text-white mb-2">Explanation:</h4>
                  <p className="text-purple-200 text-sm">{quest.solutionExplanation}</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  // Render tests tab
  const renderTestsTab = () => {
    if (!quest?.testCases) {
      return (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-white mb-2">No Test Cases Available</h3>
          <p className="text-gray-400">This quest doesn't have predefined test cases.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {quest.testCases.map((testCase, index) => (
          <div key={index} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-white">Test Case {index + 1}</h3>
              <span className="text-sm text-gray-400">
                {testCase.points || 10} points
              </span>
            </div>
            
            {testCase.description && (
              <p className="text-gray-300 mb-3">{testCase.description}</p>
            )}
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {testCase.input && (
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-2">Input:</h4>
                  <div className="bg-gray-900 p-3 rounded font-mono text-sm text-gray-300">
                    {typeof testCase.input === 'string' 
                      ? testCase.input 
                      : JSON.stringify(testCase.input, null, 2)
                    }
                  </div>
                </div>
              )}
              
              {testCase.expectedOutput && (
                <div>
                  <h4 className="text-sm font-medium text-gray-400 mb-2">Expected Output:</h4>
                  <div className="bg-gray-900 p-3 rounded font-mono text-sm text-gray-300">
                    {typeof testCase.expectedOutput === 'string' 
                      ? testCase.expectedOutput 
                      : JSON.stringify(testCase.expectedOutput, null, 2)
                    }
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Render hints tab
  const renderHintsTab = () => {
    const hints = hintsData?.data?.hints || [];

    return (
      <div className="space-y-4">
        {hints.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center">
            <Lightbulb className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-white mb-2">No Hints Used Yet</h3>
            <p className="text-gray-400 mb-4">
              Use the hint button in the code tab when you need guidance.
            </p>
            <button
              onClick={handleHintRequest}
              disabled={hintsUsed >= (quest?.maxHints || 3)}
              className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
            >
              Get First Hint
            </button>
          </div>
        ) : (
          hints.map((hint, index) => (
            <div key={index} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-yellow-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-semibold text-sm">{index + 1}</span>
                </div>
                <h3 className="text-lg font-semibold text-white">Hint {index + 1}</h3>
              </div>
              <p className="text-gray-300">{hint.content}</p>
              {hint.codeExample && (
                <div className="mt-4 bg-gray-900 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-400 mb-2">Code Example:</h4>
                  <pre className="text-gray-300 font-mono text-sm overflow-x-auto">
                    <code>{hint.codeExample}</code>
                  </pre>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    );
  };

  // Render analytics tab
  const renderAnalyticsTab = () => {
    const analytics = progress?.analytics || {};

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Time Breakdown */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-400" />
            Time Breakdown
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Reading Instructions</span>
              <span className="text-white">{formatTime(analytics.readingTime || 0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Writing Code</span>
              <span className="text-white">{formatTime(analytics.codingTime || 0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Debugging</span>
              <span className="text-white">{formatTime(analytics.debuggingTime || 0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Testing</span>
              <span className="text-white">{formatTime(analytics.testingTime || 0)}</span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-green-400" />
            Performance Metrics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Code Quality Score</span>
              <span className="text-white">{analytics.codeQuality || 'N/A'}/100</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Test Pass Rate</span>
              <span className="text-white">{analytics.testPassRate || 0}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Efficiency Score</span>
              <span className="text-white">{analytics.efficiency || 'N/A'}/10</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Learning Progress</span>
              <span className="text-white">{analytics.learningProgress || 0}%</span>
            </div>
          </div>
        </div>

        {/* Skills Development */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 lg:col-span-2">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Skills Development
          </h3>
          {analytics.skillsImproved ? (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(analytics.skillsImproved).map(([skill, improvement]) => (
                <div key={skill} className="text-center">
                  <div className="text-2xl font-bold text-white">+{improvement}%</div>
                  <div className="text-sm text-gray-400 capitalize">{skill.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400">Complete this quest to see skills development data.</p>
          )}
        </div>
      </div>
    );
  };

  // Render user notes
  const renderUserNotes = () => {
    if (!showNotes) return null;

    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-6"
      >
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-400" />
          Personal Notes
        </h3>
        <textarea
          value={userNotes}
          onChange={(e) => setUserNotes(e.target.value)}
          placeholder="Add your thoughts, insights, or reminders about this quest..."
          className="w-full h-32 bg-gray-900 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-400 resize-none focus:outline-none focus:border-blue-500"
        />
      </motion.div>
    );
  };

  // Loading state
  if (questLoading) {
    return (
      <div className="quest-detail h-full flex items-center justify-center">
        <LoadingSpinner
          size="large"
          text="Loading quest details..."
          description="Preparing your coding challenge"
        />
      </div>
    );
  }

  // Error state
  if (questError) {
    return (
      <div className="quest-detail h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">Quest Not Found</h2>
          <p className="text-gray-400 mb-6">
            The quest you're looking for doesn't exist or has been removed.
          </p>
          <button
            onClick={() => navigate('/quests')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            Back to Quest Board
          </button>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="quest-detail h-full overflow-auto"
    >
      <div className="max-w-6xl mx-auto p-6" ref={contentRef}>
        {/* Quest Header */}
        {renderQuestHeader()}

        {/* User Notes */}
        <AnimatePresence>
          {renderUserNotes()}
        </AnimatePresence>

        {/* Tab Navigation */}
        {renderTabNavigation()}

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
            {activeTab === 'instructions' && renderInstructionsTab()}
            {activeTab === 'code' && renderCodeTab()}
            {activeTab === 'tests' && renderTestsTab()}
            {activeTab === 'hints' && renderHintsTab()}
            {activeTab === 'analytics' && renderAnalyticsTab()}
          </motion.div>
        </AnimatePresence>

        {/* Related Quests */}
        {relatedQuests?.data && relatedQuests.data.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-8"
          >
            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-green-400" />
              Related Quests
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {relatedQuests.data.slice(0, 3).map((relatedQuest) => (
                <div
                  key={relatedQuest.id}
                  onClick={() => navigate(`/quests/${relatedQuest.id}`)}
                  className="bg-gray-800 border border-gray-700 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition-colors"
                >
                  <h4 className="font-semibold text-white mb-2">{relatedQuest.title}</h4>
                  <p className="text-gray-400 text-sm mb-3">{relatedQuest.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">{relatedQuest.questType}</span>
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>

      {/* Vault Reveal Modal */}
      <VaultRevealModal
        vaultItem={vaultUnlock}
        isOpen={!!vaultUnlock}
        onClose={() => setVaultUnlock(null)}
      />
    </motion.div>
  );
};

export default QuestDetail;