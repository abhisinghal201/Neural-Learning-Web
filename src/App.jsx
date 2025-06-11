import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

// Pages
import Home from './pages/Home';
import LearningSegment from './pages/LearningSegment';
import VaultArchive from './pages/VaultArchive';
import Settings from './pages/Settings';

// Components
import Navigation from './components/Navigation/Navigation';
import ErrorBoundary from './components/ErrorBoundary';
import LoadingScreen from './components/LoadingScreen';
import NeuralBackground from './components/NeuralBackground';

// Utils
import { initializeApp } from './utils/init';

// Styles
import './styles/main.css';

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState(null);

  useEffect(() => {
    const init = async () => {
      try {
        await initializeApp();
        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setInitError(error);
        setIsInitialized(true); // Allow app to load even if init fails
      }
    };

    init();
  }, []);

  if (!isInitialized) {
    return <LoadingScreen />;
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="app">
            {/* Neural Network Background */}
            <NeuralBackground />
            
            {/* Navigation */}
            <Navigation />
            
            {/* Main Content */}
            <main className="main-content">
              <AnimatePresence mode="wait">
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/learning" element={<LearningSegment />} />
                  <Route path="/learning/:phase" element={<LearningSegment />} />
                  <Route path="/learning/:phase/:week" element={<LearningSegment />} />
                  <Route path="/vault" element={<VaultArchive />} />
                  <Route path="/vault/:category" element={<VaultArchive />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </AnimatePresence>
            </main>
            
            {/* Toast Notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1a1a1a',
                  color: '#e0e0e0',
                  border: '1px solid #333',
                  borderRadius: '12px',
                  fontSize: '14px',
                  padding: '16px',
                  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
                },
                success: {
                  iconTheme: {
                    primary: '#00d4ff',
                    secondary: '#1a1a1a',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ff4757',
                    secondary: '#1a1a1a',
                  },
                },
                loading: {
                  iconTheme: {
                    primary: '#ffa502',
                    secondary: '#1a1a1a',
                  },
                },
              }}
            />
            
            {/* Initialization Error Display */}
            {initError && (
              <div className="fixed bottom-4 left-4 bg-red-900/20 border border-red-500/30 rounded-lg p-4 max-w-md">
                <div className="flex items-center gap-2 text-red-400 mb-2">
                  <span className="text-lg">⚠️</span>
                  <span className="font-medium">Initialization Warning</span>
                </div>
                <p className="text-sm text-red-300">
                  Some features may not work properly. Check your backend connection.
                </p>
              </div>
            )}
          </div>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;