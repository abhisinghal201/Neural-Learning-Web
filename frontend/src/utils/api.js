/**
 * Neural Odyssey API Client
 * 
 * Centralized API communication layer with:
 * - Request/Response interceptors
 * - Error handling and retry logic
 * - Loading states and caching
 * - TypeScript-like JSDoc annotations
 * - Request/Response logging in development
 * 
 * Author: Neural Explorer
 */

import axios from 'axios';
import toast from 'react-hot-toast';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const API_VERSION = 'v1';
const REQUEST_TIMEOUT = 30000; // 30 seconds

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/${API_VERSION}`,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor for logging and auth
apiClient.interceptors.request.use(
  (config) => {
    // Add timestamp to requests
    config.metadata = { startTime: Date.now() };
    
    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`, {
        data: config.data,
        params: config.params,
      });
    }
    
    // Add auth token if available (for future use)
    const token = localStorage.getItem('neural_odyssey_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging and global error handling
apiClient.interceptors.response.use(
  (response) => {
    // Calculate request duration
    const duration = Date.now() - response.config.metadata.startTime;
    
    // Log responses in development
    if (import.meta.env.DEV) {
      console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`, {
        status: response.status,
        data: response.data,
      });
    }
    
    return response;
  },
  (error) => {
    // Calculate request duration for failed requests
    const duration = error.config?.metadata?.startTime 
      ? Date.now() - error.config.metadata.startTime 
      : 0;
    
    // Log errors in development
    if (import.meta.env.DEV) {
      console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
      });
    }
    
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          console.warn('Bad Request:', data?.error?.message || error.message);
          break;
        case 401:
          console.warn('Unauthorized - redirecting to login');
          // Handle authentication errors
          localStorage.removeItem('neural_odyssey_token');
          // You could redirect to login page here
          break;
        case 403:
          console.warn('Forbidden:', data?.error?.message || error.message);
          toast.error('Access denied. You don\'t have permission for this action.');
          break;
        case 404:
          console.warn('Not Found:', error.config?.url);
          break;
        case 429:
          console.warn('Rate Limited');
          toast.error('Too many requests. Please slow down.');
          break;
        case 500:
          console.error('Server Error:', data?.error?.message || error.message);
          toast.error('Server error. Please try again later.');
          break;
        case 503:
          console.error('Service Unavailable');
          toast.error('Service temporarily unavailable. Please try again.');
          break;
        default:
          console.error('API Error:', data?.error?.message || error.message);
      }
    } else if (error.request) {
      // Network error
      console.error('Network Error:', error.message);
      toast.error('Network error. Please check your connection.');
    } else {
      // Other error
      console.error('Request Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

/**
 * Learning API methods
 */
export const learningApi = {
  /**
   * Get overall learning progress
   * @returns {Promise<Object>} User progress data
   */
  getProgress: () => apiClient.get('/learning/progress'),
  
  /**
   * Get phase-specific progress
   * @param {number} phase - Phase number (1-4)
   * @returns {Promise<Object>} Phase progress data
   */
  getPhaseProgress: (phase) => apiClient.get(`/learning/progress/phase/${phase}`),
  
  /**
   * Get lessons with optional filters
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Lessons data
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
   * @param {Object} progressData - Progress data
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
   * Get quest completions with filters
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Quests data
   */
  getQuests: (params = {}) => apiClient.get('/learning/quests', { params }),
  
  /**
   * Submit quest completion
   * @param {Object} questData - Quest completion data
   * @returns {Promise<Object>} Quest completion result
   */
  submitQuest: (questData) => apiClient.post('/learning/quests', questData),
  
  /**
   * Update quest completion
   * @param {number} questId - Quest completion ID
   * @param {Object} questData - Updated quest data
   * @returns {Promise<Object>} Updated quest data
   */
  updateQuest: (questId, questData) => apiClient.put(`/learning/quests/${questId}`, questData),
  
  /**
   * Get daily learning sessions
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Sessions data
   */
  getSessions: (params = {}) => apiClient.get('/learning/sessions', { params }),
  
  /**
   * Create or update daily session
   * @param {Object} sessionData - Session data
   * @returns {Promise<Object>} Session result
   */
  createSession: (sessionData) => apiClient.post('/learning/sessions', sessionData),
  
  /**
   * Update specific session
   * @param {string} date - Session date (YYYY-MM-DD)
   * @param {Object} sessionData - Updated session data
   * @returns {Promise<Object>} Updated session data
   */
  updateSession: (date, sessionData) => apiClient.put(`/learning/sessions/${date}`, sessionData),
  
  /**
   * Get spaced repetition items due for review
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Due items data
   */
  getSpacedRepetition: (params = {}) => apiClient.get('/learning/spaced-repetition', { params }),
  
  /**
   * Record spaced repetition review
   * @param {string} itemId - Item ID
   * @param {Object} reviewData - Review data
   * @returns {Promise<Object>} Review result
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
};

/**
 * Vault API methods
 */
export const vaultApi = {
  /**
   * Get all vault items with unlock status
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
   * Attempt to unlock vault item
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
   * Check and unlock eligible items
   * @returns {Promise<Object>} Newly unlocked items
   */
  checkUnlocks: () => apiClient.post('/vault/check-unlocks'),
  
  /**
   * Get vault items grouped by category
   * @returns {Promise<Object>} Categories data
   */
  getCategories: () => apiClient.get('/vault/categories'),
  
  /**
   * Get recently unlocked items
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Recent items data
   */
  getRecent: (params = {}) => apiClient.get('/vault/recent', { params }),
  
  /**
   * Get vault statistics
   * @returns {Promise<Object>} Vault statistics
   */
  getStats: () => apiClient.get('/vault/stats'),
};

/**
 * System API methods
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
  getDocs: () => apiClient.get('/'),
};

/**
 * Generic API methods
 */
export const api = {
  // HTTP methods
  get: (url, config) => apiClient.get(url, config),
  post: (url, data, config) => apiClient.post(url, data, config),
  put: (url, data, config) => apiClient.put(url, data, config),
  patch: (url, data, config) => apiClient.patch(url, data, config),
  delete: (url, config) => apiClient.delete(url, config),
  
  // Specialized methods
  learning: learningApi,
  vault: vaultApi,
  system: systemApi,
};

/**
 * Utility functions
 */

/**
 * Create a React Query key from API endpoint and parameters
 * @param {string} endpoint - API endpoint
 * @param {Object} params - Query parameters
 * @returns {Array} React Query key
 */
export const createQueryKey = (endpoint, params = {}) => {
  const baseKey = endpoint.split('/').filter(Boolean);
  if (Object.keys(params).length > 0) {
    baseKey.push(params);
  }
  return baseKey;
};

/**
 * Handle API errors with user-friendly messages
 * @param {Error} error - Axios error object
 * @param {string} defaultMessage - Default error message
 * @returns {string} User-friendly error message
 */
export const getErrorMessage = (error, defaultMessage = 'An error occurred') => {
  if (error.response?.data?.error?.message) {
    return error.response.data.error.message;
  }
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  if (error.message) {
    return error.message;
  }
  return defaultMessage;
};

/**
 * Check if error is a network error
 * @param {Error} error - Error object
 * @returns {boolean} True if network error
 */
export const isNetworkError = (error) => {
  return !error.response && error.request;
};

/**
 * Check if error is a client error (4xx)
 * @param {Error} error - Error object
 * @returns {boolean} True if client error
 */
export const isClientError = (error) => {
  return error.response?.status >= 400 && error.response?.status < 500;
};

/**
 * Check if error is a server error (5xx)
 * @param {Error} error - Error object
 * @returns {boolean} True if server error
 */
export const isServerError = (error) => {
  return error.response?.status >= 500;
};

/**
 * Retry function for failed requests
 * @param {Function} fn - Function to retry
 * @param {number} retries - Number of retries
 * @param {number} delay - Delay between retries (ms)
 * @returns {Promise} Function result
 */
export const retryRequest = async (fn, retries = 3, delay = 1000) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1 || isClientError(error)) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
    }
  }
};

/**
 * Upload file with progress tracking
 * @param {string} url - Upload URL
 * @param {FormData} formData - Form data with file
 * @param {Function} onProgress - Progress callback
 * @returns {Promise} Upload result
 */
export const uploadFile = (url, formData, onProgress) => {
  return apiClient.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
        onProgress(progress);
      }
    },
  });
};

/**
 * Download file with progress tracking
 * @param {string} url - Download URL
 * @param {string} filename - Filename for download
 * @param {Function} onProgress - Progress callback
 * @returns {Promise} Download result
 */
export const downloadFile = async (url, filename, onProgress) => {
  const response = await apiClient.get(url, {
    responseType: 'blob',
    onDownloadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
        onProgress(progress);
      }
    },
  });
  
  // Create download link
  const blob = new Blob([response.data]);
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
  
  return response;
};

// Export default api object
export default api;