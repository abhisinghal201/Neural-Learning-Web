/**
 * Neural Odyssey Lesson Detail Component
 *
 * Comprehensive lesson learning interface providing immersive content delivery,
 * progress tracking, and interactive learning elements across all lesson types.
 *
 * Features:
 * - Multi-format content rendering (theory, math, visual, coding)
 * - Real-time progress tracking and auto-save
 * - Interactive code playground integration
 * - Mathematical notation and formula rendering
 * - Rich text content with markdown support
 * - Visual content display and interaction
 * - Audio narration and speed control
 * - Note-taking and bookmark system
 * - Achievement and vault unlock notifications
 * - Adaptive navigation and recommendations
 * - Study timer and session tracking
 * - Accessibility and keyboard navigation
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Brain,
  Code,
  Eye,
  Target,
  Clock,
  Play,
  Pause,
  Square,
  RotateCcw,
  FastForward,
  Rewind,
  Volume2,
  VolumeX,
  Settings,
  Bookmark,
  BookmarkCheck,
  Star,
  CheckCircle,
  Circle,
  Award,
  Trophy,
  Lightbulb,
  Sparkles,
  Zap,
  Timer,
  Coffee,
  Moon,
  Sun,
  Maximize2,
  Minimize2,
  Share2,
  Download,
  Upload,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  MoreHorizontal,
  MessageCircle,
  Edit3,
  Save,
  X,
  Check,
  AlertCircle,
  Info,
  HelpCircle,
  ExternalLink,
  Home,
  Map,
  Compass,
  Activity,
  TrendingUp,
  BarChart3,
  PieChart,
  Users,
  Globe,
  Calendar,
  Hash,
  Tag,
  Search,
  Filter,
  SortAsc,
  Navigation,
  Layers,
  FileText,
  Image,
  Video,
  Headphones,
  Keyboard,
  Mouse,
  Smartphone,
  Monitor,
  Wifi,
  Battery,
  Signal
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import CodePlayground from '../components/CodePlayground';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const LessonDetail = () => {
  const { lessonId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  // Refs
  const contentRef = useRef(null);
  const sessionTimerRef = useRef(null);
  const autoSaveTimerRef = useRef(null);

  // State management
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showNotes, setShowNotes] = useState(false);
  const [showCodePlayground, setShowCodePlayground] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState(Date.now());
  const [sessionDuration, setSessionDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioSpeed, setAudioSpeed] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [localNotes, setLocalNotes] = useState('');
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [showAchievement, setShowAchievement] = useState(null);
  const [lessonSettings, setLessonSettings] = useState({
    fontSize: 'medium',
    theme: 'dark',
    autoplay: false,
    autoSave: true,
    showHints: true
  });

  // Fetch lesson data
  const { data: lessonData, isLoading: lessonLoading, error: lessonError } = useQuery(
    ['lesson', lessonId],
    () => api.get(`/learning/lessons/${lessonId}`),
    {
      enabled: !!lessonId,
      refetchOnWindowFocus: false,
      onSuccess: (data) => {
        const lesson = data.data;
        setCurrentProgress(lesson.completion_percentage || 0);
        setLocalNotes(lesson.notes || '');
        setIsBookmarked(lesson.is_bookmarked || false);
      }
    }
  );

  // Fetch lesson content
  const { data: contentData, isLoading: contentLoading } = useQuery(
    ['lessonContent', lessonId],
    () => api.get(`/learning/lessons/${lessonId}/content`),
    {
      enabled: !!lessonId,
      refetchOnWindowFocus: false
    }
  );

  // Fetch related lessons
  const { data: relatedLessons } = useQuery(
    ['relatedLessons', lessonId],
    () => api.get(`/learning/lessons/${lessonId}/related`),
    {
      enabled: !!lessonId,
      refetchOnWindowFocus: false
    }
  );

  // Fetch lesson navigation
  const { data: navigationData } = useQuery(
    ['lessonNavigation', lessonId],
    () => api.get(`/learning/lessons/${lessonId}/navigation`),
    {
      enabled: !!lessonId,
      refetchOnWindowFocus: false
    }
  );

  // Progress update mutation
  const updateProgressMutation = useMutation(
    (progressData) => api.put(`/learning/lessons/${lessonId}/progress`, progressData),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(['lesson', lessonId]);
        queryClient.invalidateQueries(['learningProgress']);
        
        // Check for achievements
        if (data.data.achievements?.length > 0) {
          setShowAchievement(data.data.achievements[0]);
        }

        // Check for vault unlocks
        if (data.data.vaultUnlocks?.length > 0) {
          toast.success(`🗝️ Vault item unlocked: ${data.data.vaultUnlocks[0].title}`, {
            duration: 6000
          });
        }
      },
      onError: (error) => {
        toast.error(error.response?.data?.message || 'Failed to update progress');
      }
    }
  );

  // Bookmark mutation
  const bookmarkMutation = useMutation(
    () => api.post(`/learning/lessons/${lessonId}/bookmark`),
    {
      onSuccess: () => {
        setIsBookmarked(!isBookmarked);
        toast.success(isBookmarked ? 'Bookmark removed' : 'Lesson bookmarked');
      }
    }
  );

  // Complete lesson mutation
  const completeLessonMutation = useMutation(
    (completionData) => api.post(`/learning/lessons/${lessonId}/complete`, completionData),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(['lesson', lessonId]);
        queryClient.invalidateQueries(['learningProgress']);
        toast.success('Lesson completed! 🎉');
        
        if (data.data.nextLesson) {
          setTimeout(() => {
            navigate(`/learning/lesson/${data.data.nextLesson.id}`);
          }, 2000);
        }
      }
    }
  );

  // Lesson types configuration
  const lessonTypes = {
    theory: {
      name: 'Theory',
      icon: BookOpen,
      color: 'blue',
      description: 'Conceptual understanding and knowledge'
    },
    math: {
      name: 'Mathematics',
      icon: Brain,
      color: 'green',
      description: 'Mathematical concepts and problem solving'
    },
    visual: {
      name: 'Visual',
      icon: Eye,
      color: 'purple',
      description: 'Visual demonstrations and interactive content'
    },
    coding: {
      name: 'Coding',
      icon: Code,
      color: 'orange',
      description: 'Programming exercises and implementation'
    }
  };

  // Current lesson info
  const lesson = lessonData?.data;
  const content = contentData?.data;
  const typeInfo = lesson ? lessonTypes[lesson.lesson_type] || lessonTypes.theory : null;

  // Session timer effect
  useEffect(() => {
    sessionTimerRef.current = setInterval(() => {
      setSessionDuration(Date.now() - sessionStartTime);
    }, 1000);

    return () => {
      if (sessionTimerRef.current) {
        clearInterval(sessionTimerRef.current);
      }
    };
  }, [sessionStartTime]);

  // Auto-save effect
  useEffect(() => {
    if (lessonSettings.autoSave && lesson) {
      autoSaveTimerRef.current = setInterval(() => {
        handleAutoSave();
      }, 30000); // Auto-save every 30 seconds

      return () => {
        if (autoSaveTimerRef.current) {
          clearInterval(autoSaveTimerRef.current);
        }
      };
    }
  }, [lessonSettings.autoSave, localNotes, currentProgress]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (event) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 's':
            event.preventDefault();
            handleSaveProgress();
            break;
          case 'b':
            event.preventDefault();
            bookmarkMutation.mutate();
            break;
          case 'n':
            event.preventDefault();
            setShowNotes(!showNotes);
            break;
          case 'f':
            event.preventDefault();
            setIsFullscreen(!isFullscreen);
            break;
        }
      }
      
      if (event.key === 'Escape') {
        setIsFullscreen(false);
        setShowCodePlayground(false);
      }
    };

    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, [showNotes, isFullscreen, isBookmarked]);

  // Auto-save function
  const handleAutoSave = () => {
    if (lesson && (localNotes !== lesson.notes || currentProgress !== lesson.completion_percentage)) {
      updateProgressMutation.mutate({
        completion_percentage: currentProgress,
        notes: localNotes,
        time_spent_minutes: Math.floor(sessionDuration / 60000),
        auto_save: true
      });
    }
  };

  // Save progress manually
  const handleSaveProgress = () => {
    updateProgressMutation.mutate({
      completion_percentage: currentProgress,
      notes: localNotes,
      time_spent_minutes: Math.floor(sessionDuration / 60000)
    });
    toast.success('Progress saved');
  };

  // Complete lesson
  const handleCompleteLesson = () => {
    const completionData = {
      status: 'completed',
      completion_percentage: 100,
      notes: localNotes,
      time_spent_minutes: Math.floor(sessionDuration / 60000),
      mastery_score: currentProgress >= 90 ? 0.95 : 0.8
    };

    completeLessonMutation.mutate(completionData);
  };

  // Mark as mastered
  const handleMarkAsMastered = () => {
    const masteryData = {
      status: 'mastered',
      completion_percentage: 100,
      notes: localNotes,
      time_spent_minutes: Math.floor(sessionDuration / 60000),
      mastery_score: 1.0
    };

    updateProgressMutation.mutate(masteryData);
    toast.success('Marked as mastered! 🏆');
  };

  // Navigate to lesson
  const navigateToLesson = (direction) => {
    const nav = navigationData?.data;
    if (direction === 'prev' && nav?.previousLesson) {
      navigate(`/learning/lesson/${nav.previousLesson.id}`);
    } else if (direction === 'next' && nav?.nextLesson) {
      navigate(`/learning/lesson/${nav.nextLesson.id}`);
    }
  };

  // Format time
  const formatTime = (milliseconds) => {
    const seconds = Math.floor(milliseconds / 1000);
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

  // Render lesson content based on type
  const renderLessonContent = () => {
    if (!content) return null;

    switch (lesson.lesson_type) {
      case 'coding':
        return (
          <div className="space-y-6">
            {content.description && (
              <div className="prose prose-invert max-w-none">
                <div dangerouslySetInnerHTML={{ __html: content.description }} />
              </div>
            )}
            
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Code Exercise</h3>
                <button
                  onClick={() => setShowCodePlayground(true)}
                  className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors"
                >
                  <Code className="w-4 h-4" />
                  <span>Open Code Editor</span>
                </button>
              </div>
              
              {content.starter_code && (
                <pre className="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm">
                  <code className="text-green-400">{content.starter_code}</code>
                </pre>
              )}
            </div>
          </div>
        );

      case 'math':
        return (
          <div className="space-y-6">
            <div className="prose prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: content.content }} />
            </div>
            
            {content.formulas && content.formulas.length > 0 && (
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Key Formulas</h3>
                <div className="space-y-4">
                  {content.formulas.map((formula, index) => (
                    <div key={index} className="bg-gray-900 p-4 rounded-lg">
                      <div className="text-green-400 font-mono text-lg mb-2">{formula.equation}</div>
                      <div className="text-gray-300 text-sm">{formula.description}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {content.exercises && content.exercises.length > 0 && (
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Practice Exercises</h3>
                <div className="space-y-4">
                  {content.exercises.map((exercise, index) => (
                    <div key={index} className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-white font-medium mb-2">Exercise {index + 1}</div>
                      <div className="text-gray-300">{exercise.problem}</div>
                      {exercise.hint && (
                        <details className="mt-2">
                          <summary className="text-blue-400 cursor-pointer text-sm">Show Hint</summary>
                          <div className="text-gray-400 text-sm mt-1">{exercise.hint}</div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'visual':
        return (
          <div className="space-y-6">
            <div className="prose prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: content.content }} />
            </div>
            
            {content.visualizations && content.visualizations.length > 0 && (
              <div className="space-y-4">
                {content.visualizations.map((viz, index) => (
                  <div key={index} className="bg-gray-800 border border-gray-700 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">{viz.title}</h3>
                    
                    {viz.type === 'image' && (
                      <div className="text-center">
                        <img 
                          src={viz.url} 
                          alt={viz.alt || viz.title}
                          className="max-w-full h-auto rounded-lg"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'block';
                          }}
                        />
                        <div style={{ display: 'none' }} className="text-gray-400 p-8">
                          Image placeholder: {viz.title}
                        </div>
                      </div>
                    )}
                    
                    {viz.type === 'interactive' && (
                      <div className="text-center text-gray-400 p-8 border-2 border-dashed border-gray-600 rounded-lg">
                        Interactive visualization: {viz.title}
                        <br />
                        <small>Would be rendered with D3.js or similar library</small>
                      </div>
                    )}
                    
                    {viz.description && (
                      <div className="mt-4 text-gray-300">{viz.description}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'theory':
      default:
        return (
          <div className="space-y-6">
            <div className="prose prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: content.content }} />
            </div>
            
            {content.key_concepts && content.key_concepts.length > 0 && (
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Key Concepts</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {content.key_concepts.map((concept, index) => (
                    <div key={index} className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-blue-400 font-medium mb-2">{concept.term}</div>
                      <div className="text-gray-300 text-sm">{concept.definition}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {content.takeaways && content.takeaways.length > 0 && (
              <div className="bg-blue-900 bg-opacity-50 border border-blue-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                  <Lightbulb className="w-5 h-5 text-yellow-400" />
                  <span>Key Takeaways</span>
                </h3>
                <ul className="space-y-2">
                  {content.takeaways.map((takeaway, index) => (
                    <li key={index} className="text-gray-300 flex items-start space-x-2">
                      <Check className="w-4 h-4 text-green-400 mt-1 flex-shrink-0" />
                      <span>{takeaway}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        );
    }
  };

  // Render notes panel
  const renderNotesPanel = () => (
    <AnimatePresence>
      {showNotes && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 320, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          className="bg-gray-800 border-l border-gray-700 overflow-hidden"
        >
          <div className="p-4 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Notes</h3>
              <button
                onClick={() => setShowNotes(false)}
                className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <textarea
              value={localNotes}
              onChange={(e) => setLocalNotes(e.target.value)}
              placeholder="Add your notes here..."
              className="flex-1 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-400 resize-none focus:border-blue-500 focus:outline-none"
            />
            
            <div className="mt-4 flex space-x-2">
              <button
                onClick={handleSaveProgress}
                disabled={updateProgressMutation.isLoading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-3 py-2 rounded-lg text-sm transition-colors"
              >
                {updateProgressMutation.isLoading ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Render achievement modal
  const renderAchievementModal = () => (
    <AnimatePresence>
      {showAchievement && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black bg-opacity-75 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            className="bg-gray-800 border border-gray-700 rounded-lg p-8 max-w-md w-full text-center"
          >
            <div className="w-16 h-16 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <Trophy className="w-8 h-8 text-white" />
            </div>
            
            <h3 className="text-xl font-bold text-white mb-2">Achievement Unlocked!</h3>
            <p className="text-gray-300 mb-6">{showAchievement.title}</p>
            
            <button
              onClick={() => setShowAchievement(null)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
            >
              Continue
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Loading state
  if (lessonLoading || contentLoading) {
    return (
      <div className="lesson-detail h-full">
        <LoadingSpinner
          size="large"
          text="Loading lesson..."
          description="Preparing your personalized learning experience"
        />
      </div>
    );
  }

  // Error state
  if (lessonError || !lesson) {
    return (
      <div className="lesson-detail min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">Lesson Not Found</h2>
          <p className="text-gray-400 mb-6">The lesson you're looking for doesn't exist or has been moved.</p>
          <button
            onClick={() => navigate('/learning')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors"
          >
            Back to Learning Path
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
      className={`lesson-detail ${isFullscreen ? 'fixed inset-0 z-40' : 'min-h-screen'} bg-gray-900 flex`}
    >
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-gray-800 border-b border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate(-1)}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              
              <div className="flex items-center space-x-3">
                <div className={`p-2 bg-${typeInfo.color}-600 bg-opacity-20 rounded-lg`}>
                  <typeInfo.icon className={`w-5 h-5 text-${typeInfo.color}-400`} />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">{lesson.lesson_title}</h1>
                  <div className="flex items-center space-x-2 text-sm text-gray-400">
                    <span className={`text-${typeInfo.color}-400`}>{typeInfo.name}</span>
                    <span>•</span>
                    <span>Phase {lesson.phase}, Week {lesson.week}</span>
                    <span>•</span>
                    <span>{formatTime(sessionDuration)}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Progress indicator */}
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 bg-${typeInfo.color}-400 rounded-full transition-all duration-300`}
                    style={{ width: `${currentProgress}%` }}
                  />
                </div>
                <span className="text-sm text-gray-400 w-12">{Math.round(currentProgress)}%</span>
              </div>

              {/* Action buttons */}
              <button
                onClick={bookmarkMutation.mutate}
                className={`p-2 rounded-lg transition-colors ${
                  isBookmarked ? 'text-yellow-400 bg-yellow-400 bg-opacity-20' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {isBookmarked ? <BookmarkCheck className="w-5 h-5" /> : <Bookmark className="w-5 h-5" />}
              </button>

              <button
                onClick={() => setShowNotes(!showNotes)}
                className={`p-2 rounded-lg transition-colors ${
                  showNotes ? 'text-blue-400 bg-blue-400 bg-opacity-20' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                <Edit3 className="w-5 h-5" />
              </button>

              <button
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
              >
                {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Main Content */}
          <div className="flex-1 overflow-y-auto">
            <div ref={contentRef} className="p-6 max-w-4xl mx-auto">
              {renderLessonContent()}

              {/* Learning Objectives */}
              {lesson.learning_objectives && lesson.learning_objectives.length > 0 && (
                <div className="mt-8 bg-gray-800 border border-gray-700 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Learning Objectives</h3>
                  <ul className="space-y-2">
                    {lesson.learning_objectives.map((objective, index) => (
                      <li key={index} className="text-gray-300 flex items-start space-x-2">
                        <Target className="w-4 h-4 text-blue-400 mt-1 flex-shrink-0" />
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Related Content */}
              {relatedLessons?.data && relatedLessons.data.length > 0 && (
                <div className="mt-8 bg-gray-800 border border-gray-700 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Related Lessons</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {relatedLessons.data.map((relatedLesson) => (
                      <button
                        key={relatedLesson.id}
                        onClick={() => navigate(`/learning/lesson/${relatedLesson.id}`)}
                        className="text-left p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                      >
                        <div className="font-medium text-white mb-1">{relatedLesson.title}</div>
                        <div className="text-sm text-gray-400">{relatedLesson.type} • Phase {relatedLesson.phase}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Notes Panel */}
          {renderNotesPanel()}
        </div>

        {/* Footer Controls */}
        <div className="bg-gray-800 border-t border-gray-700 p-4">
          <div className="flex items-center justify-between">
            {/* Navigation */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => navigateToLesson('prev')}
                disabled={!navigationData?.data?.previousLesson}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>

              <button
                onClick={() => navigateToLesson('next')}
                disabled={!navigationData?.data?.nextLesson}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Progress Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={handleSaveProgress}
                disabled={updateProgressMutation.isLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center space-x-2"
              >
                <Save className="w-4 h-4" />
                <span>{updateProgressMutation.isLoading ? 'Saving...' : 'Save Progress'}</span>
              </button>

              {lesson.status !== 'completed' && lesson.status !== 'mastered' && (
                <button
                  onClick={handleCompleteLesson}
                  disabled={completeLessonMutation.isLoading}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center space-x-2"
                >
                  <CheckCircle className="w-4 h-4" />
                  <span>Complete Lesson</span>
                </button>
              )}

              {lesson.status === 'completed' && (
                <button
                  onClick={handleMarkAsMastered}
                  className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors flex items-center space-x-2"
                >
                  <Trophy className="w-4 h-4" />
                  <span>Mark as Mastered</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Code Playground Modal */}
      <AnimatePresence>
        {showCodePlayground && lesson.lesson_type === 'coding' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black bg-opacity-75"
          >
            <div className="h-full flex flex-col">
              <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Code Editor</h3>
                <button
                  onClick={() => setShowCodePlayground(false)}
                  className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="flex-1">
                <CodePlayground
                  initialCode={content?.starter_code}
                  onCodeSubmit={(code, output) => {
                    setCurrentProgress(Math.min(currentProgress + 20, 100));
                    toast.success('Code executed successfully!');
                  }}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Achievement Modal */}
      {renderAchievementModal()}
    </motion.div>
  );
};

export default LessonDetail;