/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Neural Odyssey Brand Colors
        neural: {
          50: '#f0fdff',
          100: '#ccf7fe',
          200: '#9aeefc',
          300: '#5dddf9',
          400: '#00d4ff', // Primary brand color
          500: '#00abc7',
          600: '#0088a3',
          700: '#0d6b83',
          800: '#155569',
          900: '#164858',
          950: '#082f3b',
        },
        
        // Vault Categories
        vault: {
          archives: '#ffd700',     // Gold for Secret Archives 🗝️
          controversy: '#ff4757',  // Red for Controversy Files ⚔️
          beautiful: '#8b5cf6',    // Purple for Beautiful Mind 💎
        },
        
        // Quest Types
        quest: {
          theory: '#3b82f6',      // Blue for theory
          math: '#10b981',        // Green for mathematics
          visual: '#8b5cf6',      // Purple for visual projects
          coding: '#f59e0b',      // Orange for coding
          application: '#ef4444',  // Red for applications
        },
        
        // Skill Categories
        skill: {
          mathematics: '#00d4ff',
          programming: '#0099cc',
          theory: '#7b68ee',
          applications: '#ff6b9d',
          creativity: '#ffa502',
          persistence: '#ff4757',
        },
        
        // Status Colors
        status: {
          'not-started': '#6b7280',
          'in-progress': '#3b82f6',
          'completed': '#10b981',
          'mastered': '#fbbf24',
        },
      },
      
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
        display: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
      },
      
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
      },
      
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
      
      boxShadow: {
        'neural': '0 0 0 1px rgba(0, 212, 255, 0.05), 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'neural-lg': '0 0 0 1px rgba(0, 212, 255, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'neural-xl': '0 0 0 1px rgba(0, 212, 255, 0.05), 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        'glow': '0 0 20px rgba(0, 212, 255, 0.3)',
        'glow-lg': '0 0 40px rgba(0, 212, 255, 0.4)',
        'inner-glow': 'inset 0 0 20px rgba(0, 212, 255, 0.1)',
      },
      
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'neural-pulse': 'neuralPulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'neural-ping': 'neuralPing 1s cubic-bezier(0, 0, 0.2, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'shimmer': 'shimmer 2.5s ease-in-out infinite',
        'progress': 'progress 1s ease-in-out',
        'bounce-gentle': 'bounceGentle 2s infinite',
        'spin-slow': 'spin 3s linear infinite',
        'wiggle': 'wiggle 1s ease-in-out infinite',
        'typewriter': 'typewriter 3s steps(20) infinite',
      },
      
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        neuralPulse: {
          '0%, 100%': { 
            opacity: '1',
            transform: 'scale(1)',
          },
          '50%': { 
            opacity: '0.8',
            transform: 'scale(1.05)',
          },
        },
        neuralPing: {
          '75%, 100%': {
            transform: 'scale(2)',
            opacity: '0',
          },
        },
        float: {
          '0%, 100%': { 
            transform: 'translateY(0px) rotate(0deg)',
            opacity: '0.7',
          },
          '50%': { 
            transform: 'translateY(-20px) rotate(180deg)',
            opacity: '1',
          },
        },
        glow: {
          '0%': { 
            'box-shadow': '0 0 5px rgba(0, 212, 255, 0.2), 0 0 10px rgba(0, 212, 255, 0.2), 0 0 15px rgba(0, 212, 255, 0.2)',
          },
          '100%': { 
            'box-shadow': '0 0 10px rgba(0, 212, 255, 0.4), 0 0 20px rgba(0, 212, 255, 0.4), 0 0 30px rgba(0, 212, 255, 0.4)',
          },
        },
        shimmer: {
          '0%': {
            'background-position': '-1000px 0',
          },
          '100%': {
            'background-position': '1000px 0',
          },
        },
        progress: {
          '0%': { 
            transform: 'translateX(-100%)',
          },
          '100%': { 
            transform: 'translateX(0%)',
          },
        },
        bounceGentle: {
          '0%, 100%': {
            transform: 'translateY(-5%)',
            'animation-timing-function': 'cubic-bezier(0.8, 0, 1, 1)',
          },
          '50%': {
            transform: 'translateY(0)',
            'animation-timing-function': 'cubic-bezier(0, 0, 0.2, 1)',
          },
        },
        wiggle: {
          '0%, 100%': { transform: 'rotate(-3deg)' },
          '50%': { transform: 'rotate(3deg)' },
        },
        typewriter: {
          '0%': { width: '0ch' },
          '50%': { width: '20ch' },
          '100%': { width: '0ch' },
        },
      },
      
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'neural-gradient': 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)',
        'neural-gradient-dark': 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        'shimmer-gradient': 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)',
        'glass-gradient': 'linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
      },
      
      backdropBlur: {
        xs: '2px',
      },
      
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
      
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
        'colors-transform': 'color, background-color, border-color, text-decoration-color, fill, stroke, transform',
      },
      
      transitionDuration: {
        '2000': '2000ms',
        '3000': '3000ms',
      },
      
      scale: {
        '102': '1.02',
        '103': '1.03',
      },
      
      rotate: {
        '15': '15deg',
        '30': '30deg',
        '60': '60deg',
        '135': '135deg',
        '270': '270deg',
      },
      
      blur: {
        '4xl': '72px',
        '5xl': '96px',
      },
      
      brightness: {
        '25': '.25',
        '175': '1.75',
      },
      
      contrast: {
        '25': '.25',
        '175': '1.75',
      },
      
      grayscale: {
        '50': '0.5',
      },
      
      hueRotate: {
        '15': '15deg',
        '30': '30deg',
        '60': '60deg',
        '90': '90deg',
        '270': '270deg',
      },
      
      invert: {
        '25': '.25',
        '75': '.75',
      },
      
      saturate: {
        '25': '.25',
        '75': '.75',
        '175': '1.75',
      },
      
      sepia: {
        '25': '.25',
        '75': '.75',
      },
      
      screens: {
        '3xl': '1600px',
        '4xl': '1920px',
      },
    },
  },
  plugins: [
    // Custom utilities for Neural Odyssey
    function({ addUtilities, addComponents, theme }) {
      // Glass morphism utilities
      addUtilities({
        '.glass': {
          'background': 'rgba(255, 255, 255, 0.05)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.glass-dark': {
          'background': 'rgba(0, 0, 0, 0.3)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.glass-neural': {
          'background': 'rgba(0, 212, 255, 0.05)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(0, 212, 255, 0.1)',
        },
      });

      // Neural-themed components
      addComponents({
        '.btn-neural': {
          'background': 'linear-gradient(135deg, #00d4ff, #0099cc)',
          'color': 'white',
          'padding': '0.75rem 1.5rem',
          'border-radius': '0.75rem',
          'font-weight': '600',
          'transition': 'all 0.2s',
          'border': 'none',
          'cursor': 'pointer',
          '&:hover': {
            'transform': 'translateY(-2px)',
            'box-shadow': '0 10px 25px rgba(0, 212, 255, 0.3)',
          },
          '&:active': {
            'transform': 'translateY(0)',
          },
        },
        '.btn-neural-outline': {
          'background': 'transparent',
          'color': '#00d4ff',
          'padding': '0.75rem 1.5rem',
          'border-radius': '0.75rem',
          'font-weight': '600',
          'transition': 'all 0.2s',
          'border': '2px solid #00d4ff',
          'cursor': 'pointer',
          '&:hover': {
            'background': '#00d4ff',
            'color': 'white',
            'transform': 'translateY(-2px)',
          },
        },
        '.card-neural': {
          'background': 'rgba(31, 41, 55, 0.5)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(75, 85, 99, 0.3)',
          'border-radius': '1rem',
          'padding': '1.5rem',
          'transition': 'all 0.3s',
          '&:hover': {
            'border-color': 'rgba(0, 212, 255, 0.3)',
            'box-shadow': '0 10px 40px rgba(0, 0, 0, 0.3)',
          },
        },
        '.progress-neural': {
          'width': '100%',
          'height': '0.5rem',
          'background-color': 'rgba(75, 85, 99, 0.3)',
          'border-radius': '9999px',
          'overflow': 'hidden',
          '& .progress-bar': {
            'height': '100%',
            'background': 'linear-gradient(90deg, #00d4ff, #0099cc)',
            'border-radius': '9999px',
            'transition': 'width 0.3s ease',
          },
        },
        '.neural-node': {
          'position': 'relative',
          'display': 'inline-flex',
          'align-items': 'center',
          'justify-content': 'center',
          'width': '3rem',
          'height': '3rem',
          'border-radius': '50%',
          'background': 'linear-gradient(135deg, #00d4ff, #0099cc)',
          'color': 'white',
          'font-weight': '600',
          'box-shadow': '0 0 20px rgba(0, 212, 255, 0.3)',
          '&::before': {
            'content': '""',
            'position': 'absolute',
            'inset': '-2px',
            'border-radius': '50%',
            'background': 'linear-gradient(135deg, #00d4ff, #0099cc)',
            'opacity': '0.5',
            'z-index': '-1',
            'animation': 'neuralPulse 2s infinite',
          },
        },
      });

      // Text gradients
      addUtilities({
        '.text-gradient-neural': {
          'background': 'linear-gradient(135deg, #00d4ff, #0099cc)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        '.text-gradient-vault': {
          'background': 'linear-gradient(135deg, #ffd700, #ffed4e)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        '.text-gradient-rainbow': {
          'background': 'linear-gradient(135deg, #ff6b9d, #8b5cf6, #00d4ff, #10b981)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
      });

      // Scrollbar styles
      addUtilities({
        '.scrollbar-thin': {
          'scrollbar-width': 'thin',
          'scrollbar-color': '#374151 #1f2937',
        },
        '.scrollbar-none': {
          'scrollbar-width': 'none',
          '-ms-overflow-style': 'none',
          '&::-webkit-scrollbar': {
            'display': 'none',
          },
        },
        '.scrollbar-neural': {
          '&::-webkit-scrollbar': {
            'width': '8px',
          },
          '&::-webkit-scrollbar-track': {
            'background': '#1f2937',
            'border-radius': '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            'background': 'linear-gradient(135deg, #00d4ff, #0099cc)',
            'border-radius': '4px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            'background': 'linear-gradient(135deg, #0099cc, #007399)',
          },
        },
      });
    },
  ],
}