import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './styles/main.css';

// Performance monitoring
const startTime = performance.now();

// Enable React DevTools in development
if (import.meta.env.DEV) {
  import('@axe-core/react').then((axe) => {
    axe.default(React, ReactDOM, 1000);
  });
}

// Create root and render app
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Hide loading screen and show app
setTimeout(() => {
  const loadingScreen = document.getElementById('neural-loading');
  const rootElement = document.getElementById('root');
  
  if (loadingScreen) {
    loadingScreen.classList.add('hidden');
  }
  
  if (rootElement) {
    rootElement.classList.add('loaded');
  }
  
  // Log performance metrics
  const endTime = performance.now();
  console.log(`🧠 Neural Odyssey initialized in ${Math.round(endTime - startTime)}ms`);
}, 1000);

// Global error handling
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
});

// Register service worker in production
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