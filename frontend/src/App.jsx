import React, { useState, useEffect, Suspense } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { useQuery } from 'react-query';
import toast from 'react-hot-toast';

// API utilities
import { api } from './utils/api';

// Layout components
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import LoadingSpinner from './components/UI/LoadingSpinner';

// Page components (lazy loaded for code splitting)
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const LearningPath = React.lazy(() => import('./pages/LearningPath'));
const QuestBoard = React.lazy(() => import('./pages/QuestBoard'));
const VaultArchive = React.lazy(() => import('./pages/VaultArchive'));
const ProgressTracker = React.lazy(() => import('./pages/ProgressTracker'));
const Settings = React.lazy(() => import('./pages/Settings'));
const LessonDetail = React.lazy(() => import('./pages/LessonDetail'));
const QuestDetail = React.lazy(() => import('./pages/QuestDetail'));

// Animation variants
const pageVariants = {
  initial: {
    opacity: 0,
    x: 20,
  },
  in: {
    opacity: 1,
    x: 0,
  },
  out: {
    opacity: 0,
    x: -20,
  },
};

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.3,
};

// Main App component
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState('dark');
  const location = useLocation();

  // Fetch user profile and system health
  const { data: profile, isLoading: profileLoading, error: profileError } = useQuery(
    'userProfile',
    () => api.get('/learning/progress').then(res => res.data),
    {
      retry: 3,
      refetchOnWindowFocus: false,
      onError: (error) => {
        console.error('Failed to load user profile:', error);
        toast.error('Failed to load user profile. Please check your connection.');
      },
    }
  );

  const { data: healthStatus } = useQuery(
    'systemHealth',
    () => api.get('/health').then(res => res.data),
    {
      retry: 1,
      refetchInterval: 30000, // Check every 30 seconds
      refetchOnWindowFocus: false,
      onError: (error) => {
        console.error('System health check failed:', error);
      },
    }
  );

  // Check for newly unlocked vault items
  const { data: newUnlocks } = useQuery(
    'newVaultUnlocks',
    () => api.post('/vault/check-unlocks').then(res => res.data),
    {
      refetchInterval: 60000, // Check every minute
      refetchOnWindowFocus: false,
      onSuccess: (data) => {
        if (data.data?.newlyUnlocked?.length > 0) {
          data.data.newlyUnlocked.forEach(item => {
            toast.success(`🗝️ Vault item unlocked: ${item.title}`, {
              duration: 6000,
            });
          });
        }
      },
    }
  );

  // Handle theme changes
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.className = theme;
  }, [theme]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (event) => {
      // Ctrl/Cmd + K for quick search
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        // TODO: Open search modal
        console.log('Quick search triggered');
      }
      
      // Ctrl/Cmd + B for sidebar toggle
      if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
        event.preventDefault();
        setSidebarOpen(prev => !prev);
      }
      
      // Escape to close sidebar on mobile
      if (event.key === 'Escape' && sidebarOpen) {
        setSidebarOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, [sidebarOpen]);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Loading fallback component
  const PageLoadingFallback = () => (
    <div className="flex items-center justify-center min-h-[400px]">
      <LoadingSpinner size="large" text="Loading your Neural Odyssey..." />
    </div>
  );

  // Error fallback for individual routes
  const RouteErrorFallback = ({ error }) => (
    <div className="flex items-center justify-center min-h-[400px] p-4">
      <div className="text-center max-w-md">
        <div className="w-16 h-16 mx-auto mb-4 bg-red-500/10 rounded-full flex items-center justify-center">
          <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Failed to Load Page</h2>
        <p className="text-dark-300 mb-4">
          {error?.message || 'An unexpected error occurred while loading this page.'}
        </p>
        <button
          onClick={() => window.location.reload()}
          className="bg-primary-600 hover:bg-primary-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          Refresh Page
        </button>
      </div>
    </div>
  );

  // Show loading screen while profile is loading
  if (profileLoading) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center">
        <LoadingSpinner size="large" text="Initializing Neural Odyssey..." />
      </div>
    );
  }

  // Show error screen if profile failed to load
  if (profileError) {
    return (
      <div className="min-h-screen bg-dark-900 flex items-center justify-center p-4">
        <div className="text-center max-w-md">
          <div className="w-20 h-20 mx-auto mb-6 bg-red-500/10 rounded-full flex items-center justify-center">
            <svg className="w-10 h-10 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white mb-4">Connection Failed</h1>
          <p className="text-dark-300 mb-6">
            Unable to connect to the Neural Odyssey backend. Please make sure the server is running.
          </p>
          <div className="space-y-3">
            <button
              onClick={() => window.location.reload()}
              className="w-full bg-primary-600 hover:bg-primary-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
            >
              Retry Connection
            </button>
            <details className="text-left">
              <summary className="cursor-pointer text-dark-400 hover:text-dark-300 text-sm">
                Connection Details
              </summary>
              <div className="mt-3 p-3 bg-dark-800 rounded-lg text-xs text-dark-300">
                <p><strong>Backend URL:</strong> {import.meta.env.VITE_API_URL || 'http://localhost:3001'}</p>
                <p><strong>Error:</strong> {profileError?.message}</p>
                <p className="mt-2 text-dark-400">
                  Make sure to run <code className="bg-dark-700 px-1 rounded">npm run dev</code> in the backend directory.
                </p>
              </div>
            </details>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900 text-white">
      {/* Background Pattern */}
      <div className="fixed inset-0 bg-neural-pattern opacity-30 pointer-events-none" />
      
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        profile={profile?.data}
        healthStatus={healthStatus}
      />

      {/* Main Content Area */}
      <div className={`transition-all duration-300 ${sidebarOpen ? 'lg:ml-64' : 'lg:ml-16'}`}>
        {/* Header */}
        <Header
          onSidebarToggle={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
          profile={profile?.data}
          theme={theme}
          onThemeChange={setTheme}
        />

        {/* Page Content */}
        <main className="p-4 lg:p-6 pt-20 lg:pt-24 min-h-screen">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
              className="w-full"
            >
              <Suspense fallback={<PageLoadingFallback />}>
                <Routes>
                  {/* Dashboard - Default route */}
                  <Route 
                    path="/" 
                    element={<Dashboard profile={profile?.data} />} 
                  />
                  
                  {/* Learning Path */}
                  <Route 
                    path="/learning" 
                    element={<LearningPath />} 
                  />
                  <Route 
                    path="/learning/lesson/:lessonId" 
                    element={<LessonDetail />} 
                  />
                  
                  {/* Quest Board */}
                  <Route 
                    path="/quests" 
                    element={<QuestBoard />} 
                  />
                  <Route 
                    path="/quests/:questId" 
                    element={<QuestDetail />} 
                  />
                  
                  {/* Vault Archive */}
                  <Route 
                    path="/vault" 
                    element={<VaultArchive />} 
                  />
                  
                  {/* Progress Tracker */}
                  <Route 
                    path="/progress" 
                    element={<ProgressTracker profile={profile?.data} />} 
                  />
                  
                  {/* Settings */}
                  <Route 
                    path="/settings" 
                    element={<Settings theme={theme} onThemeChange={setTheme} />} 
                  />
                  
                  {/* Catch-all route */}
                  <Route 
                    path="*" 
                    element={
                      <div className="text-center py-12">
                        <div className="w-20 h-20 mx-auto mb-6 bg-dark-800 rounded-full flex items-center justify-center">
                          <svg className="w-10 h-10 text-dark-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <h1 className="text-2xl font-bold mb-4">Page Not Found</h1>
                        <p className="text-dark-300 mb-6">
                          The page you're looking for doesn't exist in the Neural Odyssey.
                        </p>
                        <Navigate to="/" replace />
                      </div>
                    } 
                  />
                </Routes>
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {/* Development Info Panel */}
      {import.meta.env.DEV && (
        <div className="fixed bottom-4 left-4 z-50">
          <details className="bg-dark-800 border border-dark-700 rounded-lg p-2 text-xs">
            <summary className="cursor-pointer text-primary-400 hover:text-primary-300">
              Dev Info
            </summary>
            <div className="mt-2 space-y-1 text-dark-300">
              <div>Route: {location.pathname}</div>
              <div>Profile: {profile ? '✅' : '❌'}</div>
              <div>Health: {healthStatus?.status || '❓'}</div>
              <div>Theme: {theme}</div>
              <div>Sidebar: {sidebarOpen ? 'Open' : 'Closed'}</div>
            </div>
          </details>
        </div>
      )}

      {/* Global Keyboard Shortcuts Help */}
      {import.meta.env.DEV && (
        <div className="fixed bottom-4 right-4 z-40">
          <details className="bg-dark-800 border border-dark-700 rounded-lg p-2 text-xs">
            <summary className="cursor-pointer text-primary-400 hover:text-primary-300">
              Shortcuts
            </summary>
            <div className="mt-2 space-y-1 text-dark-300">
              <div><kbd className="bg-dark-700 px-1 rounded">Ctrl+K</kbd> Search</div>
              <div><kbd className="bg-dark-700 px-1 rounded">Ctrl+B</kbd> Toggle Sidebar</div>
              <div><kbd className="bg-dark-700 px-1 rounded">Esc</kbd> Close Sidebar</div>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

export default App;