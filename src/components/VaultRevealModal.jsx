import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Star, 
  ExternalLink, 
  Share2, 
  BookOpen, 
  Lightbulb,
  Flame,
  Sparkles,
  Award,
  Eye,
  MessageSquare
} from 'lucide-react';
import { useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';
import Confetti from 'react-confetti';
import { api } from '../utils/api';

const VaultRevealModal = ({ vaultItem, isOpen, onClose, isNewUnlock = false }) => {
  const [showConfetti, setShowConfetti] = useState(false);
  const [userRating, setUserRating] = useState(vaultItem?.user_rating || 0);
  const [userNotes, setUserNotes] = useState(vaultItem?.user_notes || '');
  const [showNotes, setShowNotes] = useState(false);
  const [hasRead, setHasRead] = useState(vaultItem?.is_read || false);
  
  const queryClient = useQueryClient();

  // Show confetti for new unlocks
  useEffect(() => {
    if (isNewUnlock && isOpen) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);
    }
  }, [isNewUnlock, isOpen]);

  // Mark as read mutation
  const markAsReadMutation = useMutation(
    () => api.post(`/vault/items/${vaultItem.id}/read`),
    {
      onSuccess: () => {
        setHasRead(true);
        queryClient.invalidateQueries('vaultItems');
      }
    }
  );

  // Rate item mutation
  const rateItemMutation = useMutation(
    ({ rating, notes }) => api.post(`/vault/items/${vaultItem.id}/rate`, { rating, notes }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vaultItems');
        toast.success('Rating saved!');
      },
      onError: () => {
        toast.error('Failed to save rating');
      }
    }
  );

  // Auto-mark as read when opened
  useEffect(() => {
    if (isOpen && vaultItem && !hasRead) {
      markAsReadMutation.mutate();
    }
  }, [isOpen, vaultItem, hasRead]);

  // Get category styling
  const getCategoryInfo = (category) => {
    switch (category) {
      case 'secret_archives':
        return {
          name: 'Secret Archives',
          icon: '🗝️',
          color: 'text-yellow-400',
          bgGradient: 'from-yellow-500/20 to-orange-500/20',
          borderColor: 'border-yellow-500/30',
          description: 'Hidden connections and mind-blowing revelations'
        };
      case 'controversy_files':
        return {
          name: 'Controversy Files',
          icon: '⚔️',
          color: 'text-red-400',
          bgGradient: 'from-red-500/20 to-pink-500/20',
          borderColor: 'border-red-500/30',
          description: 'Drama, feuds, and heated debates in AI history'
        };
      case 'beautiful_mind':
        return {
          name: 'Beautiful Mind Collection',
          icon: '💎',
          color: 'text-purple-400',
          bgGradient: 'from-purple-500/20 to-blue-500/20',
          borderColor: 'border-purple-500/30',
          description: 'Mathematical elegance and stunning insights'
        };
      default:
        return {
          name: 'Neural Vault',
          icon: '📖',
          color: 'text-blue-400',
          bgGradient: 'from-blue-500/20 to-teal-500/20',
          borderColor: 'border-blue-500/30',
          description: 'Knowledge awaits'
        };
    }
  };

  // Handle rating submission
  const handleRatingSubmit = () => {
    if (userRating > 0) {
      rateItemMutation.mutate({ rating: userRating, notes: userNotes });
    }
  };

  // Share functionality (mock for now)
  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: vaultItem.title,
        text: vaultItem.content?.headline || vaultItem.title,
        url: window.location.href
      });
    } else {
      navigator.clipboard.writeText(vaultItem.content?.headline || vaultItem.title);
      toast.success('Copied to clipboard!');
    }
  };

  if (!isOpen || !vaultItem) return null;

  const categoryInfo = getCategoryInfo(vaultItem.category);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        {/* Confetti for new unlocks */}
        {showConfetti && (
          <Confetti
            width={window.innerWidth}
            height={window.innerHeight}
            recycle={false}
            numberOfPieces={200}
            gravity={0.3}
          />
        )}

        <motion.div
          initial={{ scale: 0.8, opacity: 0, y: 50 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.8, opacity: 0, y: 50 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="bg-gray-900 border border-gray-700 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden relative"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className={`bg-gradient-to-r ${categoryInfo.bgGradient} border-b ${categoryInfo.borderColor} relative overflow-hidden`}>
            {/* Animated background elements */}
            <div className="absolute inset-0 opacity-10">
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  rotate: [0, 180, 360] 
                }}
                transition={{ 
                  duration: 20, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
                className="absolute -top-20 -right-20 w-40 h-40 rounded-full bg-white"
              />
              <motion.div
                animate={{ 
                  scale: [1.2, 1, 1.2],
                  rotate: [360, 180, 0] 
                }}
                transition={{ 
                  duration: 15, 
                  repeat: Infinity, 
                  ease: "linear" 
                }}
                className="absolute -bottom-16 -left-16 w-32 h-32 rounded-full bg-white"
              />
            </div>

            <div className="relative p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ delay: 0.2, type: "spring", damping: 15 }}
                    className="text-6xl"
                  >
                    {categoryInfo.icon}
                  </motion.div>
                  
                  <div>
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 }}
                      className="flex items-center gap-2 mb-2"
                    >
                      <span className={`text-sm font-medium ${categoryInfo.color}`}>
                        {categoryInfo.name}
                      </span>
                      {isNewUnlock && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: 0.5, type: "spring" }}
                          className="bg-yellow-400 text-yellow-900 px-2 py-1 rounded-full text-xs font-bold flex items-center gap-1"
                        >
                          <Sparkles className="w-3 h-3" />
                          NEW UNLOCK!
                        </motion.div>
                      )}
                    </motion.div>
                    
                    <motion.h1
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="text-2xl font-bold text-white mb-2"
                    >
                      {vaultItem.title}
                    </motion.h1>
                    
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="text-gray-300 text-sm"
                    >
                      {categoryInfo.description}
                    </motion.p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleShare}
                    className="p-2 hover:bg-white/10 rounded-lg transition-colors text-white"
                  >
                    <Share2 className="w-5 h-5" />
                  </button>
                  
                  <button
                    onClick={onClose}
                    className="p-2 hover:bg-white/10 rounded-lg transition-colors text-white"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="overflow-y-auto max-h-[calc(90vh-200px)]">
            <div className="p-6">
              {/* Main Content */}
              {vaultItem.content && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                  className="space-y-6"
                >
                  {/* Headline */}
                  {vaultItem.content.headline && (
                    <div className="text-center">
                      <h2 className="text-3xl font-bold text-white mb-4 leading-tight">
                        {vaultItem.content.headline}
                      </h2>
                    </div>
                  )}

                  {/* Story */}
                  {vaultItem.content.story && (
                    <div className="prose prose-invert max-w-none">
                      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                        <div className="flex items-center gap-2 mb-4">
                          <BookOpen className="w-5 h-5 text-blue-400" />
                          <span className="font-semibold text-blue-400">The Story</span>
                        </div>
                        <p className="text-gray-300 leading-relaxed text-lg">
                          {vaultItem.content.story}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Mind Blown Section */}
                  {vaultItem.content.mindBlown && (
                    <motion.div
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: 0.8 }}
                      className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/30 rounded-xl p-6"
                    >
                      <div className="flex items-center gap-2 mb-4">
                        <motion.div
                          animate={{ rotate: [0, 10, -10, 0] }}
                          transition={{ duration: 2, repeat: Infinity }}
                        >
                          <Lightbulb className="w-5 h-5 text-yellow-400" />
                        </motion.div>
                        <span className="font-semibold text-yellow-400">Mind = Blown 🤯</span>
                      </div>
                      <p className="text-gray-200 font-medium text-lg italic">
                        "{vaultItem.content.mindBlown}"
                      </p>
                    </motion.div>
                  )}

                  {/* Deep Dive Link */}
                  {vaultItem.content.deepDive && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1.0 }}
                      className="bg-gray-800/50 rounded-xl p-4 border border-gray-700"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <ExternalLink className="w-4 h-4 text-blue-400" />
                          <span className="text-blue-400 font-medium">Want to dive deeper?</span>
                        </div>
                        <a
                          href={vaultItem.content.deepDive}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                        >
                          Explore More
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </div>

            {/* Rating & Notes Section */}
            <div className="border-t border-gray-700 bg-gray-800/30 p-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
                className="space-y-4"
              >
                {/* Rating */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-white flex items-center gap-2">
                      <Star className="w-4 h-4 text-yellow-400" />
                      Rate this revelation
                    </span>
                    {userRating > 0 && (
                      <span className="text-sm text-gray-400">
                        {userRating}/5 stars
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        onClick={() => setUserRating(star)}
                        className="p-1 hover:scale-110 transition-transform"
                      >
                        <Star
                          className={`w-6 h-6 ${
                            star <= userRating
                              ? 'fill-yellow-400 text-yellow-400'
                              : 'text-gray-600 hover:text-yellow-300'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </div>

                {/* Notes */}
                <div>
                  <button
                    onClick={() => setShowNotes(!showNotes)}
                    className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors mb-2"
                  >
                    <MessageSquare className="w-4 h-4" />
                    Add personal notes
                  </button>
                  
                  <AnimatePresence>
                    {showNotes && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <textarea
                          value={userNotes}
                          onChange={(e) => setUserNotes(e.target.value)}
                          placeholder="What did you think about this revelation? Any connections to other concepts?"
                          className="w-full bg-gray-800 border border-gray-600 rounded-lg p-3 text-gray-300 placeholder-gray-500 focus:border-blue-400 focus:outline-none resize-none"
                          rows={4}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Save Button */}
                {(userRating > 0 || userNotes.trim()) && (
                  <motion.button
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    onClick={handleRatingSubmit}
                    disabled={rateItemMutation.isLoading}
                    className="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 flex items-center gap-2"
                  >
                    <Award className="w-4 h-4" />
                    Save Feedback
                  </motion.button>
                )}
              </motion.div>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="absolute top-4 right-4 flex items-center gap-2">
            {hasRead && (
              <div className="bg-green-500/20 border border-green-500/30 rounded-full px-2 py-1 flex items-center gap-1">
                <Eye className="w-3 h-3 text-green-400" />
                <span className="text-xs text-green-400">Read</span>
              </div>
            )}
            
            {vaultItem.user_rating && (
              <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-full px-2 py-1 flex items-center gap-1">
                <Star className="w-3 h-3 text-yellow-400 fill-current" />
                <span className="text-xs text-yellow-400">{vaultItem.user_rating}</span>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default VaultRevealModal;