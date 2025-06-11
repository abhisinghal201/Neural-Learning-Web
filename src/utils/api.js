/**
 * Neural Odyssey API Utility
 * 
 * Centralized API client for communicating with the Neural Odyssey backend.
 * Provides HTTP methods, error handling, request/response interceptors,
 * and utility functions for data fetching.
 * 
 * Features:
 * - Axios-based HTTP client with interceptors
 * - Automatic request/response logging in development
 * - Error handling and retry logic
 * - Request timeout configuration
 * - Base URL and headers management
 * - Authentication token handling (if needed)
 * 
 * Author: Neural Explorer
 */

import axios from 'axios';
import toast from 'react-hot-toast';

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001/api/v1';
const API_TIMEOUT = import.meta.env.VITE_API_TIMEOUT || 30000; // 30 seconds
const isDevelopment = import.meta.env.DEV;

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request counter for loading states
let activeRequests = 0;
const requestCallbacks = new Set();

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    activeRequests++;
    requestCallbacks.forEach(callback => callback(activeRequests > 0));
    
    // Add timestamp to prevent caching issues
    if (config.method === 'get') {
      config.params = {
        ...config.params,
        _t: Date.now(),
      };
    }

    // Add authentication token if available
    const token = localStorage.getItem('neural_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Log request in development
    if (isDevelopment) {
      console.log(`🚀 API Request [${config.method?.toUpperCase()}]:`, {
        url: config.url,
        baseURL: config.baseURL,
        params: config.params,
        data: config.data,
        headers: config.headers,
      });
    }

    return config;
  },
  (error) => {
    activeRequests--;
    requestCallbacks.forEach(callback => callback(activeRequests > 0));
    
    if (isDevelopment) {
      console.error('❌ API Request Error:', error);
    }
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    activeRequests--;
    requestCallbacks.forEach(callback => callback(activeRequests > 0));

    // Log response in development
    if (isDevelopment) {
      console.log(`✅ API Response [${response.config.method?.toUpperCase()}]:`, {
        url: response.config.url,
        status: response.status,
        data: response.data,
        duration: response.config.metadata?.duration,
      });
    }

    return response;
  },
  (error) => {
    activeRequests--;
    requestCallbacks.forEach(callback => callback(activeRequests > 0));

    // Log error in development
    if (isDevelopment) {
      console.error(`❌ API Error [${error.config?.method?.toUpperCase()}]:`, {
        url: error.config?.url,
        status: error.response?.status,
        message: error.message,
        data: error.response?.data,
      });
    }

    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          toast.error(data.message || 'Invalid request');
          break;
        case 401:
          toast.error('Authentication required');
          // Clear token and redirect to login if needed
          localStorage.removeItem('neural_token');
          break;
        case 403:
          toast.error('Access denied');
          break;
        case 404:
          toast.error('Resource not found');
          break;
        case 429:
          toast.error('Too many requests. Please wait a moment.');
          break;
        case 500:
          toast.error('Server error. Please try again later.');
          break;
        case 503:
          toast.error('Service unavailable. Please check your connection.');
          break;
        default:
          toast.error(data.message || 'An unexpected error occurred');
      }
    } else if (error.request) {
      // Network error
      toast.error('Network error. Please check your internet connection.');
    } else {
      // Other error
      toast.error('An unexpected error occurred');
    }

    return Promise.reject(error);
  }
);

// Add request timing
apiClient.interceptors.request.use((config) => {
  config.metadata = { startTime: Date.now() };
  return config;
});

apiClient.interceptors.response.use(
  (response) => {
    response.config.metadata.duration = Date.now() - response.config.metadata.startTime;
    return response;
  },
  (error) => {
    if (error.config) {
      error.config.metadata.duration = Date.now() - error.config.metadata.startTime;
    }
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Generic HTTP methods
  async get(url, config = {}) {
    const response = await apiClient.get(url, config);
    return response.data;
  },

  async post(url, data = {}, config = {}) {
    const response = await apiClient.post(url, data, config);
    return response.data;
  },

  async put(url, data = {}, config = {}) {
    const response = await apiClient.put(url, data, config);
    return response.data;
  },

  async patch(url, data = {}, config = {}) {
    const response = await apiClient.patch(url, data, config);
    return response.data;
  },

  async delete(url, config = {}) {
    const response = await apiClient.delete(url, config);
    return response.data;
  },

  // Learning endpoints
  learning: {
    async getProgress(params = {}) {
      return api.get('/learning/progress', { params });
    },

    async updateProgress(lessonId, progressData) {
      return api.put(`/learning/progress/${lessonId}`, progressData);
    },

    async startSession(sessionData) {
      return api.post('/learning/sessions/start', sessionData);
    },

    async endSession(sessionId, sessionData) {
      return api.put(`/learning/sessions/${sessionId}/end`, sessionData);
    },

    async getTodaySessions() {
      return api.get('/learning/sessions/today');
    },

    async getQuests(params = {}) {
      return api.get('/learning/quests', { params });
    },

    async submitQuest(questData) {
      return api.post('/learning/quests/submit', questData);
    },

    async getReviewItems() {
      return api.get('/learning/review');
    },

    async submitReview(conceptId, reviewData) {
      return api.post(`/learning/review/${conceptId}`, reviewData);
    },

    async getAnalytics(params = {}) {
      return api.get('/learning/analytics', { params });
    },
  },

  // Vault endpoints
  vault: {
    async getItems(params = {}) {
      return api.get('/vault/items', { params });
    },

    async getItem(itemId) {
      return api.get(`/vault/items/${itemId}`);
    },

    async markAsRead(itemId) {
      return api.post(`/vault/items/${itemId}/read`);
    },

    async rateItem(itemId, ratingData) {
      return api.post(`/vault/items/${itemId}/rate`, ratingData);
    },

    async getAnalytics() {
      return api.get('/vault/analytics');
    },

    async getTimeline() {
      return api.get('/vault/timeline');
    },

    async getUpcoming() {
      return api.get('/vault/upcoming');
    },

    async checkUnlocks(unlockData) {
      return api.post('/vault/check-unlocks', unlockData);
    },
  },

  // Analytics endpoints
  analytics: {
    async getSummary() {
      return api.get('/analytics/summary');
    },

    async getDetailedStats(params = {}) {
      return api.get('/analytics/detailed', { params });
    },
  },

  // Database endpoints
  database: {
    async getStatus() {
      return api.get('/db/status');
    },

    async getStats() {
      return api.get('/db/stats');
    },
  },

  // Export endpoints
  export: {
    async exportData(format = 'json') {
      return api.post('/export/data', { format });
    },

    async exportPortfolio(options = {}) {
      return api.post('/export/portfolio', options);
    },
  },

  // Health check
  async health() {
    return api.get('/health');
  },
};

// Utility functions
export const apiUtils = {
  // Subscribe to loading state changes
  onLoadingChange(callback) {
    requestCallbacks.add(callback);
    // Return unsubscribe function
    return () => requestCallbacks.delete(callback);
  },

  // Get current loading state
  isLoading() {
    return activeRequests > 0;
  },

  // Set authentication token
  setAuthToken(token) {
    if (token) {
      localStorage.setItem('neural_token', token);
      apiClient.defaults.headers.Authorization = `Bearer ${token}`;
    } else {
      localStorage.removeItem('neural_token');
      delete apiClient.defaults.headers.Authorization;
    }
  },

  // Get authentication token
  getAuthToken() {
    return localStorage.getItem('neural_token');
  },

  // Clear authentication
  clearAuth() {
    this.setAuthToken(null);
  },

  // Update base URL
  setBaseURL(url) {
    apiClient.defaults.baseURL = url;
  },

  // Get base URL
  getBaseURL() {
    return apiClient.defaults.baseURL;
  },

  // Create cancel token
  createCancelToken() {
    return axios.CancelToken.source();
  },

  // Check if error is cancellation
  isCancel(error) {
    return axios.isCancel(error);
  },

  // Retry request with exponential backoff
  async retryRequest(requestFn, maxRetries = 3, baseDelay = 1000) {
    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx) or cancellation
        if (
          error.response?.status >= 400 && error.response?.status < 500 ||
          this.isCancel(error)
        ) {
          throw error;
        }
        
        // Don't retry on last attempt
        if (attempt === maxRetries) {
          break;
        }
        
        // Calculate delay with exponential backoff and jitter
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        
        if (isDevelopment) {
          console.log(`🔄 Retrying request (attempt ${attempt + 2}/${maxRetries + 1}) after ${delay}ms`);
        }
      }
    }
    
    throw lastError;
  },

  // Batch requests
  async batchRequests(requests, options = {}) {
    const { concurrent = 5, onProgress } = options;
    const results = [];
    const errors = [];
    
    for (let i = 0; i < requests.length; i += concurrent) {
      const batch = requests.slice(i, i + concurrent);
      const batchPromises = batch.map(async (request, index) => {
        try {
          const result = await request();
          results[i + index] = result;
          if (onProgress) {
            onProgress(i + index + 1, requests.length);
          }
          return result;
        } catch (error) {
          errors[i + index] = error;
          if (onProgress) {
            onProgress(i + index + 1, requests.length);
          }
          throw error;
        }
      });
      
      await Promise.allSettled(batchPromises);
    }
    
    return { results, errors };
  },

  // Create URL with query parameters
  createURL(endpoint, params = {}) {
    const url = new URL(endpoint, API_BASE_URL);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.append(key, value);
      }
    });
    return url.toString();
  },

  // Format error message
  formatError(error) {
    if (error.response?.data?.message) {
      return error.response.data.message;
    }
    if (error.message) {
      return error.message;
    }
    return 'An unexpected error occurred';
  },

  // Check if API is available
  async checkHealth() {
    try {
      const response = await apiClient.get('/health');
      return response.status === 200;
    } catch (error) {
      return false;
    }
  },

  // Get request statistics
  getStats() {
    return {
      activeRequests,
      baseURL: apiClient.defaults.baseURL,
      timeout: apiClient.defaults.timeout,
      hasAuthToken: !!this.getAuthToken(),
    };
  },
};

// React Query helpers
export const queryKeys = {
  // Learning keys
  learningProgress: (phase, week) => ['learning', 'progress', phase, week].filter(Boolean),
  todaySessions: () => ['learning', 'sessions', 'today'],
  quests: (params) => ['learning', 'quests', params],
  reviewItems: () => ['learning', 'review'],
  learningAnalytics: (params) => ['learning', 'analytics', params],
  
  // Vault keys
  vaultItems: (params) => ['vault', 'items', params],
  vaultItem: (id) => ['vault', 'item', id],
  vaultAnalytics: () => ['vault', 'analytics'],
  vaultTimeline: () => ['vault', 'timeline'],
  vaultUpcoming: () => ['vault', 'upcoming'],
  
  // Other keys
  analytics: (params) => ['analytics', params],
  health: () => ['health'],
  dbStatus: () => ['database', 'status'],
};

// Default export
export default api;