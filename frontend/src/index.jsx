import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { Toaster } from 'react-hot-toast';

import App from './App';
import './index.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry up to 3 times for other errors
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: false,
      onError: (error) => {
        console.error('Mutation error:', error);
      },
    },
  },
});

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to console in development
    if (import.meta.env.DEV) {
      console.error('Neural Odyssey Error:', error);
      console.error('Error Info:', errorInfo);
    }
    
    // In production, you could send this to an error reporting service
    // e.g., Sentry, LogRocket, etc.
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-dark-900 flex items-center justify-center p-4">
          <div className="max-w-md w-full">
            <div className="bg-dark-800 border border-dark-700 rounded-xl p-6 text-center">
              {/* Error Icon */}
              <div className="w-16 h-16 mx-auto mb-4 bg-red-500/10 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              
              <h1 className="text-xl font-bold text-white mb-2">
                Something went wrong
              </h1>
              
              <p className="text-dark-300 mb-6">
                An unexpected error occurred in the Neural Odyssey application. 
                Please refresh the page to try again.
              </p>
              
              <div className="space-y-3">
                <button
                  onClick={() => window.location.reload()}
                  className="w-full bg-primary-600 hover:bg-primary-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
                >
                  Refresh Page
                </button>
                
                {import.meta.env.DEV && (
                  <details className="text-left">
                    <summary className="cursor-pointer text-dark-400 hover:text-dark-300 text-sm">
                      Show Error Details (Development)
                    </summary>
                    <div className="mt-3 p-3 bg-dark-900 rounded-lg text-xs text-red-400 font-mono overflow-auto">
                      <div className="mb-2">
                        <strong>Error:</strong> {this.state.error && this.state.error.toString()}
                      </div>
                      <div>
                        <strong>Stack Trace:</strong>
                        <pre className="whitespace-pre-wrap mt-1">
                          {this.state.errorInfo.componentStack}
                        </pre>
                      </div>
                    </div>
                  </details>
                )}
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main render function
function renderApp() {
  const root = ReactDOM.createRoot(document.getElementById('root'));
  
  root.render(
    <React.StrictMode>
      <ErrorBoundary>
        <BrowserRouter>
          <QueryClientProvider client={queryClient}>
            <App />
            
            {/* React Query DevTools - only in development */}
            {import.meta.env.DEV && (
              <ReactQueryDevtools 
                initialIsOpen={false} 
                position="bottom-right"
              />
            )}
            
            {/* Global Toast Notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1e293b',
                  color: '#f8fafc',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                },
                success: {
                  iconTheme: {
                    primary: '#10b981',
                    secondary: '#f8fafc',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#f8fafc',
                  },
                },
                loading: {
                  iconTheme: {
                    primary: '#3b82f6',
                    secondary: '#f8fafc',
                  },
                },
              }}
            />
          </QueryClientProvider>
        </BrowserRouter>
      </ErrorBoundary>
    </React.StrictMode>
  );
}

// Performance monitoring
if (typeof performance !== 'undefined' && performance.mark) {
  performance.mark('neural-odyssey-render-start');
}

// Initialize app
renderApp();

// Log performance metrics in development
if (import.meta.env.DEV) {
  window.addEventListener('load', () => {
    if (typeof performance !== 'undefined' && performance.mark) {
      performance.mark('neural-odyssey-render-end');
      performance.measure('neural-odyssey-render', 'neural-odyssey-render-start', 'neural-odyssey-render-end');
      
      const renderMeasure = performance.getEntriesByName('neural-odyssey-render')[0];
      if (renderMeasure) {
        console.log(`🚀 Neural Odyssey rendered in ${renderMeasure.duration.toFixed(2)}ms`);
      }
    }
  });
  
  // Log app version and build info
  console.log(`
    🧠 Neural Odyssey Frontend
    ────────────────────────────
    Version: ${__APP_VERSION__ || 'development'}
    Built: ${__BUILD_TIME__ || 'unknown'}
    Environment: ${import.meta.env.MODE}
    React Query: ${import.meta.env.DEV ? 'DevTools Enabled' : 'Production'}
    ────────────────────────────
    Ready to start your AI/ML learning journey!
  `);
}

// Service Worker registration (if available)
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}

// Global error handler for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  
  // Prevent the default browser behavior
  event.preventDefault();
  
  // Show user-friendly error message
  if (window.toast) {
    window.toast.error('An unexpected error occurred. Please try again.');
  }
});

// Global error handler for uncaught errors
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
  
  // Show user-friendly error message for critical errors
  if (event.error && event.error.name !== 'ChunkLoadError') {
    if (window.toast) {
      window.toast.error('Something went wrong. Please refresh the page.');
    }
  }
});

// Network status monitoring
window.addEventListener('online', () => {
  console.log('🌐 Network connection restored');
  if (window.toast) {
    window.toast.success('Connection restored!');
  }
});

window.addEventListener('offline', () => {
  console.log('📵 Network connection lost');
  if (window.toast) {
    window.toast.error('Connection lost. Some features may not work.');
  }
});

// Development hot reload message
if (import.meta.hot) {
  import.meta.hot.on('vite:beforeUpdate', () => {
    console.log('🔄 Hot reloading Neural Odyssey...');
  });
}

// Export query client for external access
window.__NEURAL_ODYSSEY_QUERY_CLIENT__ = queryClient;