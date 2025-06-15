/**
 * Neural Odyssey Loading Spinner Component
 *
 * Animated loading spinner with multiple variants and customization options.
 * Supports different sizes, custom text, and neural-themed animations.
 * Used throughout the application for loading states and transitions.
 *
 * Features:
 * - Multiple size variants (small, medium, large, xl)
 * - Customizable loading text and descriptions
 * - Neural network-inspired animations
 * - Progress indicator support
 * - Accessible design with proper ARIA labels
 * - Theme-aware colors and styling
 * - Overlay and inline display modes
 *
 * Author: Neural Explorer
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Loader, Zap, Sparkles } from 'lucide-react';

const LoadingSpinner = ({
  size = 'medium',
  text = 'Loading...',
  description = null,
  variant = 'default',
  showProgress = false,
  progress = 0,
  overlay = false,
  className = '',
  textClassName = '',
  spinnerColor = 'text-blue-400',
  backgroundColor = 'bg-gray-900',
  ...props
}) => {
  // Size configurations
  const sizeConfig = {
    small: {
      spinner: 'w-6 h-6',
      container: 'gap-2',
      text: 'text-sm',
      padding: 'p-3'
    },
    medium: {
      spinner: 'w-8 h-8',
      container: 'gap-3',
      text: 'text-base',
      padding: 'p-4'
    },
    large: {
      spinner: 'w-12 h-12',
      container: 'gap-4',
      text: 'text-lg',
      padding: 'p-6'
    },
    xl: {
      spinner: 'w-16 h-16',
      container: 'gap-4',
      text: 'text-xl',
      padding: 'p-8'
    }
  };

  const config = sizeConfig[size] || sizeConfig.medium;

  // Animation variants for different spinner types
  const spinnerVariants = {
    default: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }
    },
    pulse: {
      scale: [1, 1.2, 1],
      opacity: [0.5, 1, 0.5],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut"
      }
    },
    neural: {
      rotate: [0, 180, 360],
      scale: [1, 1.1, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    },
    dots: {
      scale: [0.8, 1.2, 0.8],
      transition: {
        duration: 0.6,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  // Text animation variants
  const textVariants = {
    initial: { opacity: 0, y: 10 },
    animate: { 
      opacity: 1, 
      y: 0,
      transition: {
        delay: 0.2,
        duration: 0.3,
        ease: "easeOut"
      }
    }
  };

  // Container animation for overlay mode
  const overlayVariants = {
    initial: { opacity: 0 },
    animate: { 
      opacity: 1,
      transition: {
        duration: 0.2,
        ease: "easeOut"
      }
    }
  };

  // Render different spinner types based on variant
  const renderSpinner = () => {
    switch (variant) {
      case 'neural':
        return (
          <motion.div
            variants={spinnerVariants.neural}
            animate="neural"
            className={`${config.spinner} ${spinnerColor} relative`}
          >
            <Brain className="w-full h-full" />
            <motion.div
              animate={{
                rotate: 360,
                scale: [0.8, 1.2, 0.8],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "linear"
              }}
              className="absolute inset-0 border-2 border-blue-400 border-t-transparent rounded-full"
            />
          </motion.div>
        );

      case 'pulse':
        return (
          <motion.div
            variants={spinnerVariants.pulse}
            animate="pulse"
            className={`${config.spinner} ${spinnerColor}`}
          >
            <Sparkles className="w-full h-full" />
          </motion.div>
        );

      case 'dots':
        return (
          <div className={`flex space-x-1 ${config.spinner.includes('w-6') ? 'w-8' : config.spinner.includes('w-8') ? 'w-12' : config.spinner.includes('w-12') ? 'w-16' : 'w-20'}`}>
            {[0, 1, 2].map((dot) => (
              <motion.div
                key={dot}
                variants={spinnerVariants.dots}
                animate="dots"
                transition={{
                  ...spinnerVariants.dots.transition,
                  delay: dot * 0.2
                }}
                className={`w-2 h-2 bg-blue-400 rounded-full ${size === 'small' ? 'w-1.5 h-1.5' : size === 'large' ? 'w-3 h-3' : size === 'xl' ? 'w-4 h-4' : ''}`}
              />
            ))}
          </div>
        );

      case 'zap':
        return (
          <motion.div
            animate={{
              rotate: 360,
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className={`${config.spinner} ${spinnerColor} relative`}
          >
            <Zap className="w-full h-full" />
            <motion.div
              animate={{
                opacity: [0.3, 1, 0.3],
              }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-sm -z-10"
            />
          </motion.div>
        );

      default:
        return (
          <motion.div
            variants={spinnerVariants.default}
            animate="default"
            className={`${config.spinner} ${spinnerColor}`}
          >
            <Loader className="w-full h-full" />
          </motion.div>
        );
    }
  };

  // Render progress bar if enabled
  const renderProgress = () => {
    if (!showProgress) return null;

    return (
      <div className="w-full max-w-xs">
        <div className="flex justify-between items-center mb-1">
          <span className={`${config.text} font-medium text-gray-300`}>
            {Math.round(progress)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-gradient-to-r from-blue-400 to-blue-500 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          />
        </div>
      </div>
    );
  };

  // Main loading content
  const loadingContent = (
    <div 
      className={`flex flex-col items-center justify-center ${config.container} ${config.padding} ${className}`}
      role="status"
      aria-live="polite"
      aria-label={text}
      {...props}
    >
      {/* Spinner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        {renderSpinner()}
      </motion.div>

      {/* Text */}
      {text && (
        <motion.div
          variants={textVariants}
          initial="initial"
          animate="animate"
          className="text-center"
        >
          <div className={`${config.text} font-medium text-white ${textClassName}`}>
            {text}
          </div>
          {description && (
            <div className={`${size === 'small' ? 'text-xs' : 'text-sm'} text-gray-400 mt-1`}>
              {description}
            </div>
          )}
        </motion.div>
      )}

      {/* Progress Bar */}
      {renderProgress()}

      {/* Neural dots animation */}
      {variant === 'neural' && (
        <motion.div
          className="flex space-x-1 mt-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          {[0, 1, 2, 3, 4].map((dot) => (
            <motion.div
              key={dot}
              className="w-1 h-1 bg-blue-400 rounded-full"
              animate={{
                opacity: [0.2, 1, 0.2],
                scale: [0.8, 1.2, 0.8],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: dot * 0.1,
                ease: "easeInOut"
              }}
            />
          ))}
        </motion.div>
      )}
    </div>
  );

  // Return overlay version if requested
  if (overlay) {
    return (
      <motion.div
        variants={overlayVariants}
        initial="initial"
        animate="animate"
        className={`fixed inset-0 ${backgroundColor} bg-opacity-90 backdrop-blur-sm flex items-center justify-center z-50`}
      >
        <div className="bg-gray-800 bg-opacity-95 rounded-xl border border-gray-700 shadow-2xl">
          {loadingContent}
        </div>
      </motion.div>
    );
  }

  // Return inline version
  return loadingContent;
};

export default LoadingSpinner;