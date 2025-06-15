import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Code, 
  BookOpen, 
  Target, 
  Wrench, 
  Clock, 
  Star, 
  Trophy, 
  Play, 
  CheckCircle,
  AlertCircle,
  Lightbulb,
  Award,
  ChevronDown,
  ChevronRight,
  ExternalLink
} from 'lucide-react';
import { useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';
import { api } from '../utils/api';

const QuestCard = ({ quest, onStart, className = '' }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showHints, setShowHints] = useState(false);
  const queryClient = useQueryClient();

  // Get quest type icon and styling
  const getQuestTypeInfo = (type) => {
    switch (type) {
      case 'coding_exercise':
        return {
          icon: Code,
          label: 'Coding Exercise',
          color: 'text-blue-400',
          bgColor: 'bg-blue-400/10',
          borderColor: 'border-blue-400/30'
        };
      case 'implementation_project':
        return {
          icon: Wrench,
          label: 'Implementation Project',
          color: 'text-green-400',
          bgColor: 'bg-green-400/10',
          borderColor: 'border-green-400/30'
        };
      case 'theory_quiz':
        return {
          icon: BookOpen,
          label: 'Theory Quiz',
          color: 'text-purple-400',
          bgColor: 'bg-purple-400/10',
          borderColor: 'border-purple-400/30'
        };
      case 'practical_application':
        return {
          icon: Target,
          label: 'Practical Application',
          color: 'text-orange-400',
          bgColor: 'bg-orange-400/10',
          borderColor: 'border-orange-400/30'
        };
      default:
        return {
          icon: Target,
          label: 'Quest',
          color: 'text-gray-400',
          bgColor: 'bg-gray-400/10',
          borderColor: 'border-gray-400/30'
        };
    }
  };

  // Get difficulty stars
  const getDifficultyStars = (level) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`w-3 h-3 ${
          i < level 
            ? 'fill-yellow-400 text-yellow-400' 
            : 'text-gray-600'
        }`}
      />
    ));
  };

  // Get status styling
  const getStatusInfo = (status) => {
    switch (status) {
      case 'completed':
        return {
          icon: CheckCircle,
          label: 'Completed',
          color: 'text-green-400',
          bgColor: 'bg-green-400/10'
        };
      case 'mastered':
        return {
          icon: Trophy,
          label: 'Mastered',
          color: 'text-yellow-400',
          bgColor: 'bg-yellow-400/10'
        };
      case 'attempted':
        return {
          icon: AlertCircle,
          label: 'Attempted',
          color: 'text-orange-400',
          bgColor: 'bg-orange-400/10'
        };
      default:
        return {
          icon: Play,
          label: 'Available',
          color: 'text-blue-400',
          bgColor: 'bg-blue-400/10'
        };
    }
  };

  // Format estimated time
  const formatTime = (minutes) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  // Submit quest mutation
  const submitQuestMutation = useMutation(
    (questData) => api.post('/learning/quests/submit', questData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('quests');
        queryClient.invalidateQueries('learningProgress');
        toast.success('Quest submitted successfully!');
      },
      onError: (error) => {
        toast.error('Failed to submit quest: ' + error.message);
      }
    }
  );

  const typeInfo = getQuestTypeInfo(quest.type);
  const statusInfo = getStatusInfo(quest.status);
  const TypeIcon = typeInfo.icon;
  const StatusIcon = statusInfo.icon;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`quest-card bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden hover:border-gray-600 transition-all duration-300 ${className}`}
    >
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${typeInfo.bgColor} ${typeInfo.borderColor} border`}>
              <TypeIcon className={`w-5 h-5 ${typeInfo.color}`} />
            </div>
            <div>
              <h3 className="font-semibold text-white text-lg leading-tight">
                {quest.title}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <span className={`text-sm ${typeInfo.color}`}>
                  {typeInfo.label}
                </span>
                <span className="text-gray-400 text-sm">•</span>
                <span className="text-gray-400 text-sm">
                  Phase {quest.phase} - Week {quest.week}
                </span>
              </div>
            </div>
          </div>

          {/* Status Badge */}
          <div className={`flex items-center gap-1 px-2 py-1 rounded-lg ${statusInfo.bgColor}`}>
            <StatusIcon className={`w-3 h-3 ${statusInfo.color}`} />
            <span className={`text-xs font-medium ${statusInfo.color}`}>
              {statusInfo.label}
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          {quest.description}
        </p>

        {/* Quest Metadata */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            {/* Difficulty */}
            <div className="flex items-center gap-1">
              <span className="text-gray-400 text-xs">Difficulty:</span>
              <div className="flex items-center gap-1">
                {getDifficultyStars(quest.difficulty_level)}
              </div>
            </div>

            {/* Estimated Time */}
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-gray-400" />
              <span className="text-gray-400 text-xs">
                {formatTime(quest.estimated_time_minutes)}
              </span>
            </div>

            {/* Points Reward */}
            <div className="flex items-center gap-1">
              <Award className="w-3 h-3 text-yellow-400" />
              <span className="text-yellow-400 text-xs">
                {quest.difficulty_level * 15}pts
              </span>
            </div>
          </div>

          {/* Expand Button */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-gray-700 rounded-lg transition-colors"
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>

        {/* Learning Objectives */}
        {quest.learning_objectives && quest.learning_objectives.length > 0 && (
          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
              <Target className="w-3 h-3" />
              Learning Objectives
            </h4>
            <ul className="space-y-1">
              {quest.learning_objectives.slice(0, isExpanded ? undefined : 2).map((objective, index) => (
                <li key={index} className="text-xs text-gray-400 flex items-start gap-2">
                  <span className="text-blue-400 mt-1">•</span>
                  <span>{objective}</span>
                </li>
              ))}
              {!isExpanded && quest.learning_objectives.length > 2 && (
                <li className="text-xs text-gray-500">
                  +{quest.learning_objectives.length - 2} more objectives...
                </li>
              )}
            </ul>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onStart && onStart(quest)}
            disabled={submitQuestMutation.isLoading}
            className="flex-1 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            {quest.status === 'completed' || quest.status === 'mastered' ? 'Review Quest' : 'Start Quest'}
          </button>

          {quest.hints && quest.hints.length > 0 && (
            <button
              onClick={() => setShowHints(!showHints)}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg text-sm transition-colors flex items-center gap-1"
            >
              <Lightbulb className="w-4 h-4" />
              Hints
            </button>
          )}

          {quest.resources && quest.resources.length > 0 && (
            <button className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg text-sm transition-colors">
              <ExternalLink className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-700"
          >
            <div className="p-4 space-y-4">
              {/* Starter Code Preview */}
              {quest.starter_code && (
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <Code className="w-3 h-3" />
                    Starter Code Preview
                  </h4>
                  <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                    <pre className="text-xs text-gray-300 font-mono">
                      {quest.starter_code.split('\n').slice(0, 10).join('\n')}
                      {quest.starter_code.split('\n').length > 10 && '\n...'}
                    </pre>
                  </div>
                </div>
              )}

              {/* Test Cases */}
              {quest.test_cases && quest.test_cases.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <CheckCircle className="w-3 h-3" />
                    Test Cases ({quest.test_cases.length})
                  </h4>
                  <div className="text-xs text-gray-400">
                    Your solution will be validated against {quest.test_cases.length} test cases
                  </div>
                </div>
              )}

              {/* Tags */}
              {quest.tags && quest.tags.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Tags</h4>
                  <div className="flex flex-wrap gap-1">
                    {quest.tags.map((tag, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Prerequisites */}
              {quest.prerequisites && quest.prerequisites.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                    <AlertCircle className="w-3 h-3" />
                    Prerequisites
                  </h4>
                  <ul className="space-y-1">
                    {quest.prerequisites.map((prereq, index) => (
                      <li key={index} className="text-xs text-gray-400 flex items-center gap-2">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        {prereq}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hints Panel */}
      <AnimatePresence>
        {showHints && quest.hints && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-700 bg-yellow-500/5"
          >
            <div className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="w-4 h-4 text-yellow-400" />
                <h4 className="text-sm font-medium text-yellow-400">Helpful Hints</h4>
              </div>
              <div className="space-y-2">
                {quest.hints.map((hint, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <span className="text-yellow-400 text-sm font-bold mt-0.5">
                      {index + 1}.
                    </span>
                    <p className="text-sm text-gray-300">{hint}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Completion Stats (if completed) */}
      {(quest.status === 'completed' || quest.status === 'mastered') && quest.completed_at && (
        <div className="px-4 pb-4">
          <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-4">
                <span className="text-gray-400">
                  Completed: {new Date(quest.completed_at).toLocaleDateString()}
                </span>
                {quest.time_to_complete_minutes && (
                  <span className="text-gray-400">
                    Time: {formatTime(quest.time_to_complete_minutes)}
                  </span>
                )}
                {quest.attempts_count && (
                  <span className="text-gray-400">
                    Attempts: {quest.attempts_count}
                  </span>
                )}
              </div>
              {quest.status === 'mastered' && (
                <div className="flex items-center gap-1 text-yellow-400">
                  <Trophy className="w-3 h-3" />
                  <span className="text-xs font-medium">Mastered</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default QuestCard;