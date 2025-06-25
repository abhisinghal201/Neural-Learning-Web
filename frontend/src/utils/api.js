/**
 * Enhanced Neural Odyssey API Client
 * 
 * Comprehensive API client that leverages ALL backend capabilities:
 * - Complete learning endpoints with session management
 * - Spaced repetition system with SM-2 algorithm
 * - Knowledge graph connections and analytics
 * - Vault management with unlock conditions
 * - Quest submissions with rich metadata
 * - Analytics and insights
 * - Streak tracking and achievements
 * 
 * Author: Neural Explorer
 */

import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const API_VERSION = '/api/v1';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging and authentication
apiClient.interceptors.request.use(
  (config) => {
    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
    }
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling and logging
apiClient.interceptors.response.use(
  (response) => {
    // Log responses in development
    if (import.meta.env.DEV) {
      console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, response.data);
    }
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response?.status === 429) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }
    
    if (error.response?.status >= 500) {
      throw new Error('Server error. Please check your connection and try again.');
    }
    
    throw error;
  }
);

/**
 * Learning API methods - Complete implementation
 */
export const learningApi = {
  /**
   * Get overall learning progress with filters
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Progress data
   */
  getProgress: (params = {}) => apiClient.get('/learning/progress', { params }),
  
  /**
   * Get phase-specific progress
   * @param {number} phase - Phase number
   * @param {Object} params - Additional query parameters
   * @returns {Promise<Object>} Phase progress data
   */
  getPhaseProgress: (phase, params = {}) => 
    apiClient.get(`/learning/progress/phase/${phase}`, { params }),
  
  /**
   * Get all lessons with filtering
   * @param {Object} params - Query parameters (phase, week, status, type, limit, offset)
   * @returns {Promise<Object>} Lessons data with pagination
   */
  getLessons: (params = {}) => apiClient.get('/learning/lessons', { params }),
  
  /**
   * Get specific lesson details
   * @param {string} lessonId - Lesson ID
   * @returns {Promise<Object>} Lesson details
   */
  getLesson: (lessonId) => apiClient.get(`/learning/lessons/${lessonId}`),
  
  /**
   * Update lesson progress
   * @param {string} lessonId - Lesson ID
   * @param {Object} progressData - Progress update data
   * @returns {Promise<Object>} Updated lesson data
   */
  updateLessonProgress: (lessonId, progressData) => 
    apiClient.put(`/learning/lessons/${lessonId}/progress`, progressData),
  
  /**
   * Mark lesson as completed
   * @param {string} lessonId - Lesson ID
   * @param {Object} completionData - Completion data
   * @returns {Promise<Object>} Completion result
   */
  completeLesson: (lessonId, completionData) => 
    apiClient.post(`/learning/lessons/${lessonId}/complete`, completionData),
  
  /**
   * Get quest/project completions with filters
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Quests data
   */
  getQuests: (params = {}) => apiClient.get('/learning/quests', { params }),
  
  /**
   * Submit quest completion with rich metadata
   * @param {Object} questData - Complete quest submission data
   * @returns {Promise<Object>} Submission result with mentor feedback
   */
  submitQuest: (questData) => apiClient.post('/learning/quests', questData),
  
  /**
   * Update existing quest completion
   * @param {string} questId - Quest completion ID
   * @param {Object} questData - Updated quest data
   * @returns {Promise<Object>} Updated quest data
   */
  updateQuest: (questId, questData) => apiClient.put(`/learning/quests/${questId}`, questData),
  
  /**
   * Get daily learning sessions with comprehensive data
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Sessions data
   */
  getSessions: (params = {}) => apiClient.get('/learning/sessions', { params }),
  
  /**
   * Get today's sessions with analytics
   * @returns {Promise<Object>} Today's session data
   */
  getTodaySessions: () => apiClient.get('/learning/sessions/today'),
  
  /**
   * Create new learning session with full tracking
   * @param {Object} sessionData - Complete session data
   * @returns {Promise<Object>} Session result
   */
  createSession: (sessionData) => apiClient.post('/learning/sessions', sessionData),
  
  /**
   * Update specific session with rich metadata
   * @param {string} date - Session date (YYYY-MM-DD)
   * @param {Object} sessionData - Updated session data
   * @returns {Promise<Object>} Updated session data
   */
  updateSession: (date, sessionData) => apiClient.put(`/learning/sessions/${date}`, sessionData),
  
  /**
   * Get spaced repetition items due for review
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Due items data with SM-2 metadata
   */
  getSpacedRepetition: (params = {}) => apiClient.get('/learning/spaced-repetition', { params }),
  
  /**
   * Record spaced repetition review with SM-2 algorithm
   * @param {string} itemId - Item ID
   * @param {Object} reviewData - Review data with quality score
   * @returns {Promise<Object>} Review result with next review date
   */
  reviewSpacedRepetition: (itemId, reviewData) => 
    apiClient.post(`/learning/spaced-repetition/${itemId}/review`, reviewData),
  
  /**
   * Get knowledge graph connections
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Knowledge graph data
   */
  getKnowledgeGraph: (params = {}) => apiClient.get('/learning/knowledge-graph', { params }),
  
  /**
   * Add knowledge connection
   * @param {Object} connectionData - Connection data
   * @returns {Promise<Object>} Connection result
   */
  addKnowledgeConnection: (connectionData) => 
    apiClient.post('/learning/knowledge-graph', connectionData),
  
  /**
   * Get next lesson recommendation
   * @returns {Promise<Object>} Next lesson recommendation
   */
  getNextLesson: () => apiClient.get('/learning/next-lesson'),
  
  /**
   * Get comprehensive learning analytics
   * @param {Object} params - Query parameters (timeframe, etc.)
   * @returns {Promise<Object>} Analytics data
   */
  getAnalytics: (params = {}) => apiClient.get('/learning/analytics', { params }),
  
  /**
   * Get current learning streak with history
   * @returns {Promise<Object>} Streak data
   */
  getStreak: () => apiClient.get('/learning/streak'),
  
  /**
   * Reset learning progress (with confirmation)
   * @param {Object} confirmationData - Reset confirmation data
   * @returns {Promise<Object>} Reset result
   */
  resetProgress: (confirmationData) => apiClient.post('/learning/reset-progress', confirmationData),
};

/**
 * Vault API methods - Complete implementation
 */
export const vaultApi = {
  /**
   * Get all vault items with unlock status and filters
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Vault items data
   */
  getItems: (params = {}) => apiClient.get('/vault/items', { params }),
  
  /**
   * Get specific vault item details
   * @param {string} itemId - Item ID
   * @returns {Promise<Object>} Vault item details
   */
  getItem: (itemId) => apiClient.get(`/vault/items/${itemId}`),
  
  /**
   * Attempt to unlock vault item with condition checking
   * @param {string} itemId - Item ID
   * @returns {Promise<Object>} Unlock result
   */
  unlockItem: (itemId) => apiClient.post(`/vault/unlock/${itemId}`),
  
  /**
   * Get unlocked vault items (user's collection)
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Archive data
   */
  getArchive: (params = {}) => apiClient.get('/vault/archive', { params }),
  
  /**
   * Get unlock conditions for an item
   * @param {string} itemId - Item ID
   * @returns {Promise<Object>} Unlock conditions
   */
  getUnlockConditions: (itemId) => apiClient.get(`/vault/unlock-conditions/${itemId}`),
  
  /**
   * Check and unlock eligible items automatically
   * @returns {Promise<Object>} Newly unlocked items
   */
  checkUnlocks: () => apiClient.post('/vault/check-unlocks'),
  
  /**
   * Get vault items grouped by category
   * @returns {Promise<Object>} Categories data
   */
  getCategories: () => apiClient.get('/vault/categories'),
  
  /**
   * Get vault items in specific category
   * @param {string} category - Category name
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Category items data
   */
  getCategoryItems: (category, params = {}) => 
    apiClient.get(`/vault/categories/${category}`, { params }),
  
  /**
   * Get recently unlocked items
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Recent items data
   */
  getRecent: (params = {}) => apiClient.get('/vault/recent', { params }),
  
  /**
   * Get user's favorite vault items
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Favorite items data
   */
  getFavorites: (params = {}) => apiClient.get('/vault/favorites', { params }),
  
  /**
   * Get comprehensive vault statistics
   * @returns {Promise<Object>} Vault statistics
   */
  getStatistics: () => apiClient.get('/vault/statistics'),
  
  /**
   * Mark vault item as read
   * @param {string} itemId - Item ID
   * @param {Object} readData - Read tracking data
   * @returns {Promise<Object>} Read result
   */
  markAsRead: (itemId, readData = {}) => 
    apiClient.post(`/vault/items/${itemId}/read`, readData),
  
  /**
   * Rate a vault item
   * @param {string} itemId - Item ID
   * @param {Object} ratingData - Rating data
   * @returns {Promise<Object>} Rating result
   */
  rateItem: (itemId, ratingData) => 
    apiClient.post(`/vault/items/${itemId}/rate`, ratingData),
  
  /**
   * Toggle favorite status
   * @param {string} itemId - Item ID
   * @returns {Promise<Object>} Favorite toggle result
   */
  toggleFavorite: (itemId) => apiClient.post(`/vault/items/${itemId}/favorite`),
  
  /**
   * Get personalized vault recommendations
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Recommendations data
   */
  getRecommendations: (params = {}) => apiClient.get('/vault/recommendations', { params }),
  
  /**
   * Get vault unlock timeline
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Timeline data
   */
  getTimeline: (params = {}) => apiClient.get('/vault/timeline', { params }),
  
  /**
   * Search vault items
   * @param {Object} params - Search parameters
   * @returns {Promise<Object>} Search results
   */
  searchItems: (params = {}) => apiClient.get('/vault/search', { params }),
  
  /**
   * Get detailed vault analytics
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Analytics data
   */
  getAnalytics: (params = {}) => apiClient.get('/vault/analytics', { params }),
  
  /**
   * Reload vault items from configuration (development)
   * @returns {Promise<Object>} Reload result
   */
  reloadItems: () => apiClient.post('/vault/reload'),
};

/**
 * System and utility API methods
 */
export const systemApi = {
  /**
   * Get system health status
   * @returns {Promise<Object>} Health status
   */
  getHealth: () => axios.get(`${API_BASE_URL}/health`),
  
  /**
   * Get API documentation
   * @returns {Promise<Object>} API documentation
   */
  getDocs: () => apiClient.get('/docs'),
  
  /**
   * Export user portfolio and progress
   * @param {Object} exportOptions - Export configuration
   * @returns {Promise<Object>} Export data
   */
  exportPortfolio: (exportOptions = {}) => apiClient.post('/export/portfolio', exportOptions),
  
  /**
   * Backup database
   * @returns {Promise<Object>} Backup result
   */
  backupDatabase: () => apiClient.post('/system/backup'),
  
  /**
   * Get system statistics
   * @returns {Promise<Object>} System stats
   */
  getSystemStats: () => apiClient.get('/system/stats'),
};

/**
 * Analytics API methods - Extended functionality
 */
export const analyticsApi = {
  /**
   * Get comprehensive dashboard analytics
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Dashboard analytics
   */
  getDashboard: (params = {}) => apiClient.get('/analytics/dashboard', { params }),
  
  /**
   * Get learning patterns analysis
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Learning patterns data
   */
  getLearningPatterns: (params = {}) => apiClient.get('/analytics/patterns', { params }),
  
  /**
   * Get performance insights
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Performance insights
   */
  getPerformanceInsights: (params = {}) => apiClient.get('/analytics/performance', { params }),
  
  /**
   * Get learning velocity analysis
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Velocity analysis
   */
  getVelocityAnalysis: (params = {}) => apiClient.get('/analytics/velocity', { params }),
  
  /**
   * Get focus and productivity metrics
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Focus metrics
   */
  getFocusMetrics: (params = {}) => apiClient.get('/analytics/focus', { params }),
  
  /**
   * Get skill progression analysis
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Skill progression data
   */
  getSkillProgression: (params = {}) => apiClient.get('/analytics/skills', { params }),
};

/**
 * Quick access methods for common operations
 */
export const quickActions = {
  /**
   * Start a quick learning session
   * @param {string} sessionType - Type of session
   * @param {Object} options - Session options
   * @returns {Promise<Object>} Session start result
   */
  startSession: async (sessionType = 'math', options = {}) => {
    const sessionData = {
      session_type: sessionType,
      energy_level: options.energy || 8,
      mood_before: options.mood || 'focused',
      goals: options.goals || [],
      target_duration_minutes: options.duration || 25,
      start_time: new Date().toISOString(),
      ...options
    };
    
    return learningApi.createSession(sessionData);
  },
  
  /**
   * Complete current session with metrics
   * @param {string} date - Session date
   * @param {Object} metrics - Session completion metrics
   * @returns {Promise<Object>} Session completion result
   */
  completeSession: async (date, metrics = {}) => {
    const sessionData = {
      end_time: new Date().toISOString(),
      actual_duration_minutes: metrics.duration || 25,
      focus_score: metrics.focus || 8,
      mood_after: metrics.mood || 'accomplished',
      session_notes: metrics.notes || '',
      goal_completion_rate: metrics.goalCompletion || 1.0,
      ...metrics
    };
    
    return learningApi.updateSession(date, sessionData);
  },
  
  /**
   * Submit quest with automatic metadata
   * @param {Object} questData - Quest submission data
   * @returns {Promise<Object>} Quest submission result
   */
  submitQuest: async (questData) => {
    const enrichedData = {
      completed_at: new Date().toISOString(),
      status: questData.status || 'completed',
      time_to_complete_minutes: questData.timeSpent || 0,
      attempts_count: questData.attempts || 1,
      self_reflection: questData.reflection || '',
      ...questData
    };
    
    return learningApi.submitQuest(enrichedData);
  },
  
  /**
   * Review spaced repetition item
   * @param {string} itemId - Item ID
   * @param {number} quality - Quality score (0-5)
   * @param {string} notes - Optional review notes
   * @returns {Promise<Object>} Review result
   */
  reviewItem: async (itemId, quality, notes = '') => {
    return learningApi.reviewSpacedRepetition(itemId, {
      quality_score: quality,
      review_notes: notes,
      reviewed_at: new Date().toISOString()
    });
  },
  
  /**
   * Check for new vault unlocks
   * @returns {Promise<Object>} New unlocks result
   */
  checkVaultUnlocks: async () => {
    return vaultApi.checkUnlocks();
  },
  
  /**
   * Get today's learning summary
   * @returns {Promise<Object>} Today's summary
   */
  getTodaySummary: async () => {
    const [sessions, progress, reviews] = await Promise.all([
      learningApi.getTodaySessions(),
      learningApi.getProgress({ timeframe: 'today' }),
      learningApi.getSpacedRepetition({ due_today: true })
    ]);
    
    return {
      sessions: sessions.data,
      progress: progress.data,
      reviews: reviews.data
    };
  }
};

/**
 * Default export with all API methods
 */
const api = {
  learning: learningApi,
  vault: vaultApi,
  system: systemApi,
  analytics: analyticsApi,
  quick: quickActions,
  
  // Legacy support - keep existing method signatures
  get: (url, config) => apiClient.get(url, config),
  post: (url, data, config) => apiClient.post(url, data, config),
  put: (url, data, config) => apiClient.put(url, data, config),
  delete: (url, config) => apiClient.delete(url, config),
  patch: (url, data, config) => apiClient.patch(url, data, config),
};

export default api;

// Named exports for convenience
export {
  learningApi,
  vaultApi,
  systemApi,
  analyticsApi,
  quickActions,
  apiClient
};

// Export the full api object
export { api };