#!/usr/bin/env node

/**
 * Neural Odyssey Database Initialization Script
 *
 * This script:
 * 1. Creates the SQLite database with the complete schema
 * 2. Populates initial learning content from the ML path structure
 * 3. Sets up vault items with unlock conditions
 * 4. Creates sample user profile and initial progress
 * 5. Initializes spaced repetition system
 * 6. Sets up knowledge graph connections
 *
 * Run: node scripts/init-db.js
 */

const sqlite3 = require('sqlite3').verbose()
const fs = require('fs')
const path = require('path')

// File paths
const DB_PATH = path.join(__dirname, '../data/user-progress.sqlite')
const SCHEMA_PATH = path.join(__dirname, '../data/schema.sql')
const DATA_DIR = path.join(__dirname, '../data')

// Learning path structure based on the Neural Odyssey curriculum
const LEARNING_CURRICULUM = {
  phase1: {
    title: 'Mathematical Foundations and Historical Context',
    duration: 'Months 1-3',
    weeks: [
      {
        week: 1,
        title: 'Linear Algebra Through the Lens of Data',
        lessons: [
          {
            id: 'linear_algebra_intro',
            title: 'Introduction to Linear Algebra',
            type: 'theory'
          },
          {
            id: 'vectors_and_matrices',
            title: 'Vectors and Matrices',
            type: 'math'
          },
          {
            id: 'geometric_transformations',
            title: 'Geometric Transformations',
            type: 'visual'
          },
          {
            id: 'matrix_operations_implementation',
            title: 'Matrix Operations from Scratch',
            type: 'coding'
          }
        ]
      },
      {
        week: 2,
        title: 'Calculus as the Engine of Learning',
        lessons: [
          {
            id: 'calculus_foundations',
            title: 'Calculus Foundations for ML',
            type: 'theory'
          },
          {
            id: 'derivatives_geometric',
            title: 'Derivatives and Geometric Intuition',
            type: 'math'
          },
          {
            id: 'gradient_visualization',
            title: 'Gradient Visualization',
            type: 'visual'
          },
          {
            id: 'optimization_from_scratch',
            title: 'Build Gradient Descent from Scratch',
            type: 'coding'
          }
        ]
      },
      {
        week: 3,
        title: 'Probability and Statistics',
        lessons: [
          {
            id: 'probability_foundations',
            title: 'Probability Theory Foundations',
            type: 'theory'
          },
          { id: 'bayesian_thinking', title: 'Bayesian Thinking', type: 'math' },
          {
            id: 'statistical_distributions',
            title: 'Statistical Distributions',
            type: 'visual'
          },
          {
            id: 'monte_carlo_methods',
            title: 'Monte Carlo Methods Implementation',
            type: 'coding'
          }
        ]
      },
      {
        week: 4,
        title: 'Information Theory',
        lessons: [
          {
            id: 'entropy_concepts',
            title: 'Entropy and Information',
            type: 'theory'
          },
          {
            id: 'mutual_information',
            title: 'Mutual Information',
            type: 'math'
          },
          {
            id: 'information_visualization',
            title: 'Information Theory Visualization',
            type: 'visual'
          },
          {
            id: 'compression_algorithms',
            title: 'Data Compression Algorithms',
            type: 'coding'
          }
        ]
      }
    ]
  },
  phase2: {
    title: 'Core Machine Learning',
    duration: 'Months 4-6',
    weeks: [
      {
        week: 5,
        title: 'Supervised Learning Foundations',
        lessons: [
          {
            id: 'learning_theory',
            title: 'What Does It Mean to Learn?',
            type: 'theory'
          },
          {
            id: 'bias_variance_tradeoff',
            title: 'Bias-Variance Tradeoff',
            type: 'math'
          },
          {
            id: 'overfitting_visualization',
            title: 'Overfitting Visualization',
            type: 'visual'
          },
          {
            id: 'cross_validation_implementation',
            title: 'Cross-Validation from Scratch',
            type: 'coding'
          }
        ]
      },
      {
        week: 6,
        title: 'Linear Models and Regression',
        lessons: [
          {
            id: 'linear_regression_theory',
            title: 'Linear Regression Theory',
            type: 'theory'
          },
          {
            id: 'least_squares_derivation',
            title: 'Least Squares Derivation',
            type: 'math'
          },
          {
            id: 'regression_visualization',
            title: 'Regression Visualization',
            type: 'visual'
          },
          {
            id: 'linear_regression_scratch',
            title: 'Linear Regression from Scratch',
            type: 'coding'
          }
        ]
      },
      {
        week: 7,
        title: 'Neural Network Foundations',
        lessons: [
          {
            id: 'perceptron_history',
            title: 'The Perceptron and AI History',
            type: 'theory'
          },
          {
            id: 'backpropagation_math',
            title: 'Backpropagation Mathematics',
            type: 'math'
          },
          {
            id: 'neural_network_visualization',
            title: 'Neural Network Visualization',
            type: 'visual'
          },
          {
            id: 'neural_network_scratch',
            title: 'Neural Network from Scratch',
            type: 'coding'
          }
        ]
      },
      {
        week: 8,
        title: 'Gradient Descent Deep Dive',
        lessons: [
          {
            id: 'optimization_landscape',
            title: 'Optimization Landscapes',
            type: 'theory'
          },
          {
            id: 'gradient_descent_variants',
            title: 'Gradient Descent Variants',
            type: 'math'
          },
          {
            id: 'optimization_visualization',
            title: 'Optimization Visualization',
            type: 'visual'
          },
          {
            id: 'optimizer_comparison',
            title: 'Optimizer Comparison Implementation',
            type: 'coding'
          }
        ]
      }
    ]
  },
  phase3: {
    title: 'Advanced Topics',
    duration: 'Months 7-9',
    weeks: [
      {
        week: 9,
        title: 'Deep Learning Architectures',
        lessons: [
          {
            id: 'deep_learning_history',
            title: 'Deep Learning Revolution',
            type: 'theory'
          },
          {
            id: 'activation_functions',
            title: 'Activation Functions',
            type: 'math'
          },
          {
            id: 'architecture_visualization',
            title: 'Architecture Visualization',
            type: 'visual'
          },
          {
            id: 'deep_network_implementation',
            title: 'Deep Network Implementation',
            type: 'coding'
          }
        ]
      },
      {
        week: 10,
        title: 'Convolutional Neural Networks',
        lessons: [
          {
            id: 'cnn_intuition',
            title: 'CNN Intuition and Biology',
            type: 'theory'
          },
          {
            id: 'convolution_mathematics',
            title: 'Convolution Mathematics',
            type: 'math'
          },
          {
            id: 'feature_map_visualization',
            title: 'Feature Map Visualization',
            type: 'visual'
          },
          { id: 'cnn_from_scratch', title: 'CNN from Scratch', type: 'coding' }
        ]
      },
      {
        week: 11,
        title: 'Attention Mechanisms',
        lessons: [
          {
            id: 'attention_revolution',
            title: 'The Attention Revolution',
            type: 'theory'
          },
          {
            id: 'attention_mathematics',
            title: 'Attention Mathematics',
            type: 'math'
          },
          {
            id: 'attention_visualization',
            title: 'Attention Visualization',
            type: 'visual'
          },
          {
            id: 'transformer_implementation',
            title: 'Transformer Implementation',
            type: 'coding'
          }
        ]
      },
      {
        week: 12,
        title: 'Phase Integration and Mastery',
        lessons: [
          {
            id: 'integration_project',
            title: 'Integration Project',
            type: 'project'
          },
          {
            id: 'concept_mastery_assessment',
            title: 'Concept Mastery Assessment',
            type: 'assessment'
          },
          {
            id: 'knowledge_synthesis',
            title: 'Knowledge Synthesis',
            type: 'theory'
          },
          {
            id: 'portfolio_creation',
            title: 'Portfolio Creation',
            type: 'practical'
          }
        ]
      }
    ]
  }
}

// Vault items configuration with detailed unlock conditions
const VAULT_ITEMS = {
  mathematical_foundations: [
    {
      item_id: 'secret_eigenvalue_magic',
      title: '🗝️ The Secret of Eigenvalue Magic',
      description:
        "Discover how Google's PageRank algorithm is essentially one massive eigenvector calculation",
      category: 'Mathematical Insights',
      item_type: 'secret',
      rarity: 'rare',
      unlock_conditions: {
        phases_completed: [1],
        min_understanding_score: 4.0
      },
      content_preview:
        'The mathematical secret that powers the entire internet...',
      content_full: `# The Secret of Eigenvalue Magic

Did you know that every time you search on Google, you're witnessing eigenvalue decomposition in action? The PageRank algorithm, which revolutionized web search, is fundamentally based on finding the dominant eigenvector of the web's link matrix.

## The Hidden Connection

When Larry Page and Sergey Brin created Google, they realized that the web's link structure could be represented as a massive matrix. Each webpage is a node, and each link is an edge with a weight. The PageRank of a page is its corresponding element in the dominant eigenvector of this matrix.

## The Mathematics

The PageRank vector π satisfies: π = αMπ + (1-α)v

Where:
- M is the column-stochastic matrix of the web
- α is the damping factor (usually 0.85)
- v is the personalization vector

This is essentially finding the dominant eigenvalue (which equals 1) and its corresponding eigenvector.

## The Impact

This single mathematical insight created a $280 billion company and changed how humanity accesses information. Linear algebra isn't just academic—it's the foundation of the digital age.

Every day, Google performs eigenvalue computations on matrices with billions of dimensions. The mathematics you're learning isn't just theory—it's the engine that powers the modern world.`
    },
    {
      item_id: 'calculus_everywhere',
      title: '💎 Calculus is Everywhere',
      description:
        'See how calculus powers every AI system, from Netflix recommendations to self-driving cars',
      category: 'Mathematical Insights',
      item_type: 'insight',
      rarity: 'uncommon',
      unlock_conditions: {
        lessons_completed: ['calculus_foundations', 'derivatives_geometric'],
        min_study_time_hours: 5
      },
      content_preview: 'Every AI breakthrough is powered by calculus...',
      content_full: `# Calculus Powers Everything

## Netflix's 5 Billion Daily Calculations

Netflix performs over 5 billion gradient descent steps per day to optimize their recommendation system. Every time you see a movie suggestion, calculus determined it was optimal for you.

## Tesla's Self-Driving Revolution

Every Tesla on the road performs thousands of partial derivative calculations per second:
- Object detection: Convolutional neural networks with backpropagation
- Path planning: Optimization with constraints
- Decision making: Reinforcement learning with policy gradients

## The Universal Pattern

Whether it's:
- GPT-4 generating text (gradient-based training)
- Medical diagnosis (optimization of neural networks)
- Financial trading (stochastic calculus)
- Weather prediction (partial differential equations)

Calculus is the universal language of optimization and change.

## Your Journey

Every calculus concept you master brings you closer to understanding how AI systems actually work. You're not just learning math—you're learning the language of intelligence itself.`
    }
  ],
  ai_history: [
    {
      item_id: 'ai_winter_lessons',
      title: '⚔️ Lessons from the AI Winters',
      description:
        'Learn from the AI winters of the 1970s and 1980s - why AI failed and how it came back stronger',
      category: 'AI History',
      item_type: 'story',
      rarity: 'epic',
      unlock_conditions: {
        phases_completed: [1, 2],
        min_understanding_score: 3.5,
        min_streak_days: 7
      },
      content_preview: 'The dark periods that nearly killed AI research...',
      content_full: `# Lessons from the AI Winters

## The First AI Winter (1974-1980)

The promises were grand: by 1975, machines would have human-level intelligence. Instead, AI research hit a wall.

### What Went Wrong:
- **Computational Limits**: The hardware couldn't handle the ambitious algorithms
- **Overpromising**: Researchers promised the moon but delivered pebbles
- **Funding Cuts**: The British Lighthill Report and US DARPA cuts devastated research

### The Lesson: Understanding limitations is as important as understanding capabilities

## The Second AI Winter (1987-1993)

Expert systems promised to capture human expertise in rules. Companies spent billions, but the systems were brittle and couldn't adapt.

### What Went Wrong:
- **Brittleness**: Rule-based systems broke when faced with unexpected inputs
- **Knowledge Acquisition**: The "knowledge bottleneck" - experts couldn't articulate their intuition
- **Maintenance Nightmare**: Every new scenario required new rules

### The Lesson: Flexibility and learning from data trump rigid rule systems

## The Phoenix Rises (1993-2012)

What changed everything:
- **Better Data**: The internet provided massive datasets
- **Computational Power**: Moore's Law caught up to algorithmic needs
- **New Approaches**: Statistical learning replaced symbolic reasoning

## Today's Lessons

As we enter the age of large language models and AGI claims, remember:

1. **Manage Expectations**: Every breakthrough comes with limitations
2. **Focus on Fundamentals**: Mathematical foundations outlast technological hype
3. **Learn from Data**: The most robust systems learn and adapt
4. **Embrace Uncertainty**: Probabilistic reasoning beats deterministic rules

## Your Advantage

You're learning AI during its golden age, but with the wisdom of its failures. You understand both the power and the limitations. This makes you a more thoughtful, effective AI practitioner.

The winters taught us humility. The springs taught us possibility. You're equipped with both.`
    }
  ],
  implementation_tools: [
    {
      item_id: 'neural_network_debugger',
      title: '🔧 Neural Network Debugger Tool',
      description:
        'A powerful debugging tool for understanding what goes wrong in neural network training',
      category: 'Implementation Tools',
      item_type: 'tool',
      rarity: 'uncommon',
      unlock_conditions: {
        quests_completed: ['neural_network_scratch'],
        min_understanding_score: 3.0
      },
      content_preview: 'Debug neural networks like a pro...',
      content_full: `# Neural Network Debugger Tool

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class NeuralNetworkDebugger:
    """
    A comprehensive debugging tool for neural networks that helps identify
    common training problems and provides actionable insights.
    """
    
    def __init__(self, model):
        self.model = model
        self.training_history = []
        self.gradient_history = []
        self.activation_history = []
        
    def analyze_gradients(self, epoch: int) -> Dict[str, Any]:
        """Analyze gradient flow and identify vanishing/exploding gradients."""
        analysis = {
            'epoch': epoch,
            'gradient_norms': [],
            'vanishing_layers': [],
            'exploding_layers': []
        }
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                analysis['gradient_norms'].append(grad_norm)
                
                if grad_norm < 1e-6:
                    analysis['vanishing_layers'].append(i)
                elif grad_norm > 100:
                    analysis['exploding_layers'].append(i)
        
        return analysis
    
    def analyze_activations(self, x) -> Dict[str, Any]:
        """Analyze activation distributions to detect dead neurons."""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'activation' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        _ = self.model(x)
        
        # Analyze activations
        analysis = {}
        for name, activation in activations.items():
            zero_fraction = (activation == 0).float().mean().item()
            analysis[name] = {
                'zero_fraction': zero_fraction,
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'dead_neurons': zero_fraction > 0.5
            }
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return analysis
    
    def diagnose_training_problems(self) -> List[str]:
        """Provide diagnostic suggestions based on training history."""
        suggestions = []
        
        if len(self.training_history) < 2:
            return ["Need more training history to diagnose problems"]
        
        recent_losses = self.training_history[-10:]
        
        # Check for convergence issues
        if all(loss > recent_losses[0] * 0.99 for loss in recent_losses[1:]):
            suggestions.append("Loss plateaued - try reducing learning rate or adding regularization")
        
        # Check for instability
        loss_variance = np.var(recent_losses)
        if loss_variance > np.mean(recent_losses):
            suggestions.append("Training unstable - try reducing learning rate or gradient clipping")
        
        # Check for overfitting
        if hasattr(self, 'val_history') and len(self.val_history) > 5:
            train_trend = np.polyfit(range(5), self.training_history[-5:], 1)[0]
            val_trend = np.polyfit(range(5), self.val_history[-5:], 1)[0]
            
            if train_trend < 0 and val_trend > 0:
                suggestions.append("Overfitting detected - add regularization or reduce model complexity")
        
        return suggestions
    
    def visualize_training(self):
        """Create comprehensive training visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history, label='Training Loss')
        if hasattr(self, 'val_history'):
            axes[0, 0].plot(self.val_history, label='Validation Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Gradient norms
        if self.gradient_history:
            grad_norms = [np.mean(epoch_grads['gradient_norms']) for epoch_grads in self.gradient_history]
            axes[0, 1].plot(grad_norms)
            axes[0, 1].set_title('Average Gradient Norm')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
        
        # Learning rate schedule
        if hasattr(self, 'lr_history'):
            axes[1, 0].plot(self.lr_history)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        # Activation statistics
        if self.activation_history:
            dead_neuron_fractions = [np.mean([layer_stats['zero_fraction'] 
                                            for layer_stats in epoch_stats.values()]) 
                                   for epoch_stats in self.activation_history]
            axes[1, 1].plot(dead_neuron_fractions)
            axes[1, 1].set_title('Dead Neuron Fraction')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Fraction of Dead Neurons')
        
        plt.tight_layout()
        plt.show()

# Usage Example
debugger = NeuralNetworkDebugger(your_model)

# During training loop:
for epoch in range(num_epochs):
    # ... training code ...
    
    # Debug after each epoch
    grad_analysis = debugger.analyze_gradients(epoch)
    activation_analysis = debugger.analyze_activations(sample_batch)
    
    debugger.gradient_history.append(grad_analysis)
    debugger.activation_history.append(activation_analysis)
    
    # Get suggestions
    suggestions = debugger.diagnose_training_problems()
    if suggestions:
        print(f"Epoch {epoch} suggestions: {suggestions}")

# Visualize training progress
debugger.visualize_training()
\`\`\`

## Key Features

1. **Gradient Analysis**: Detects vanishing and exploding gradients
2. **Activation Monitoring**: Identifies dead neurons and activation patterns
3. **Training Diagnostics**: Provides actionable suggestions for training issues
4. **Comprehensive Visualization**: All-in-one training dashboard

## When to Use

- Neural network training isn't converging
- Suspecting vanishing/exploding gradients
- Need to optimize training hyperparameters
- Debugging activation functions and layer configurations

This tool has saved countless hours of debugging for neural network practitioners. Use it whenever training doesn't go as expected!`
    }
  ]
}

console.log('🚀 Starting Neural Odyssey Database Initialization...\n')

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true })
  console.log('📁 Created data directory')
}

// Remove existing database if it exists
if (fs.existsSync(DB_PATH)) {
  fs.unlinkSync(DB_PATH)
  console.log('🗑️  Removed existing database')
}

// Create new database
const db = new sqlite3.Database(DB_PATH, err => {
  if (err) {
    console.error('❌ Error creating database:', err.message)
    process.exit(1)
  }
  console.log('✅ Created new SQLite database')
})

// Initialize database schema
async function initializeSchema () {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(SCHEMA_PATH)) {
      reject(new Error(`Schema file not found: ${SCHEMA_PATH}`))
      return
    }

    const schemaSql = fs.readFileSync(SCHEMA_PATH, 'utf8')

    db.exec(schemaSql, err => {
      if (err) {
        reject(err)
      } else {
        console.log('✅ Database schema initialized')
        resolve()
      }
    })
  })
}

// Create user profile
async function createUserProfile () {
  return new Promise((resolve, reject) => {
    const sql = `
            INSERT INTO user_profile (
                id, name, email, learning_goal, current_phase, current_week,
                total_study_time_minutes, streak_days, max_streak,
                preferences, learning_style, created_at, updated_at
            ) VALUES (1, 'Neural Explorer', 'explorer@neural-odyssey.local', 
                     'Become AI/ML Expert', 1, 1, 0, 0, 0, 
                     '{"notifications": true, "difficulty": "balanced"}', 
                     'visual_kinesthetic', datetime('now'), datetime('now'))
        `

    db.run(sql, err => {
      if (err) {
        reject(err)
      } else {
        console.log('✅ Created user profile')
        resolve()
      }
    })
  })
}

// Populate learning progress
async function populateLearningProgress () {
  return new Promise((resolve, reject) => {
    const lessons = []

    // Generate lessons from curriculum
    Object.keys(LEARNING_CURRICULUM).forEach(phaseKey => {
      const phase = LEARNING_CURRICULUM[phaseKey]
      const phaseNumber = parseInt(phaseKey.replace('phase', ''))

      phase.weeks.forEach(week => {
        week.lessons.forEach(lesson => {
          lessons.push({
            lesson_id: lesson.id,
            lesson_title: lesson.title,
            lesson_type: lesson.type,
            phase: phaseNumber,
            week: week.week,
            status: 'not_started',
            understanding_score: null,
            confidence_level: null,
            time_spent_minutes: 0,
            attempts_count: 0,
            last_accessed: null,
            notes: '',
            difficulty_rating: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          })
        })
      })
    })

    // Insert lessons
    const stmt = db.prepare(`
            INSERT INTO learning_progress (
                lesson_id, lesson_title, lesson_type, phase, week,
                status, understanding_score, confidence_level,
                time_spent_minutes, attempts_count, last_accessed,
                notes, difficulty_rating, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `)

    let completed = 0
    lessons.forEach(lesson => {
      stmt.run(
        [
          lesson.lesson_id,
          lesson.lesson_title,
          lesson.lesson_type,
          lesson.phase,
          lesson.week,
          lesson.status,
          lesson.understanding_score,
          lesson.confidence_level,
          lesson.time_spent_minutes,
          lesson.attempts_count,
          lesson.last_accessed,
          lesson.notes,
          lesson.difficulty_rating,
          lesson.created_at,
          lesson.updated_at
        ],
        err => {
          if (err) {
            reject(err)
            return
          }

          completed++
          if (completed === lessons.length) {
            stmt.finalize()
            console.log(`✅ Populated ${lessons.length} learning lessons`)
            resolve()
          }
        }
      )
    })
  })
}

// Populate vault items
async function populateVaultItems () {
  return new Promise((resolve, reject) => {
    const items = []

    // Generate vault items from configuration
    Object.keys(VAULT_ITEMS).forEach(category => {
      VAULT_ITEMS[category].forEach(item => {
        items.push({
          ...item,
          unlock_conditions: JSON.stringify(item.unlock_conditions),
          is_unlocked: 0,
          unlocked_at: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      })
    })

    // Insert vault items
    const stmt = db.prepare(`
            INSERT INTO vault_items (
                item_id, title, description, category, item_type, rarity,
                unlock_conditions, content_preview, content_full,
                is_unlocked, unlocked_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `)

    let completed = 0
    items.forEach(item => {
      stmt.run(
        [
          item.item_id,
          item.title,
          item.description,
          item.category,
          item.item_type,
          item.rarity,
          item.unlock_conditions,
          item.content_preview,
          item.content_full,
          item.is_unlocked,
          item.unlocked_at,
          item.created_at,
          item.updated_at
        ],
        err => {
          if (err) {
            reject(err)
            return
          }

          completed++
          if (completed === items.length) {
            stmt.finalize()
            console.log(`✅ Populated ${items.length} vault items`)
            resolve()
          }
        }
      )
    })
  })
}

// Create sample daily session
async function createSampleSession () {
  return new Promise((resolve, reject) => {
    const today = new Date().toISOString().split('T')[0]

    const sql = `
            INSERT INTO daily_sessions (
                date, study_time_minutes, quests_completed, lessons_completed,
                focus_score, notes, created_at, updated_at
            ) VALUES (?, 0, 0, 0, 100, 'Starting my Neural Odyssey journey!', 
                     datetime('now'), datetime('now'))
        `

    db.run(sql, [today], err => {
      if (err) {
        reject(err)
      } else {
        console.log('✅ Created sample daily session')
        resolve()
      }
    })
  })
}

// Add some initial knowledge graph connections
async function populateKnowledgeGraph () {
  return new Promise((resolve, reject) => {
    const connections = [
      {
        source: 'linear_algebra_intro',
        target: 'vectors_and_matrices',
        type: 'prerequisite',
        strength: 0.9,
        description:
          'Linear algebra introduction provides foundation for vectors and matrices'
      },
      {
        source: 'vectors_and_matrices',
        target: 'geometric_transformations',
        type: 'builds_on',
        strength: 0.8,
        description:
          'Geometric transformations build upon vector and matrix operations'
      },
      {
        source: 'calculus_foundations',
        target: 'derivatives_geometric',
        type: 'prerequisite',
        strength: 0.9,
        description:
          'Calculus foundations required for understanding derivatives'
      },
      {
        source: 'derivatives_geometric',
        target: 'optimization_from_scratch',
        type: 'applies_in',
        strength: 0.85,
        description:
          'Derivative concepts directly apply to optimization algorithms'
      }
    ]

    const stmt = db.prepare(`
            INSERT INTO knowledge_graph (
                source_lesson_id, target_lesson_id, connection_type,
                strength, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        `)

    let completed = 0
    connections.forEach(conn => {
      stmt.run(
        [conn.source, conn.target, conn.type, conn.strength, conn.description],
        err => {
          if (err) {
            reject(err)
            return
          }

          completed++
          if (completed === connections.length) {
            stmt.finalize()
            console.log(
              `✅ Created ${connections.length} knowledge graph connections`
            )
            resolve()
          }
        }
      )
    })
  })
}

// Write vault items to JSON file for easy editing
async function writeVaultItemsJson () {
  const vaultItemsPath = path.join(DATA_DIR, 'vault-items.json')

  try {
    fs.writeFileSync(vaultItemsPath, JSON.stringify(VAULT_ITEMS, null, 2))
    console.log('✅ Created vault-items.json configuration file')
  } catch (error) {
    console.error('❌ Error writing vault-items.json:', error.message)
  }
}

// Main initialization function
async function initializeDatabase () {
  try {
    await initializeSchema()
    await createUserProfile()
    await populateLearningProgress()
    await populateVaultItems()
    await createSampleSession()
    await populateKnowledgeGraph()
    await writeVaultItemsJson()

    console.log('\n🎉 Database initialization completed successfully!')
    console.log('\n📊 Summary:')

    // Get final statistics
    db.get('SELECT COUNT(*) as count FROM learning_progress', (err, row) => {
      if (!err) console.log(`   📚 ${row.count} learning lessons`)
    })

    db.get('SELECT COUNT(*) as count FROM vault_items', (err, row) => {
      if (!err) console.log(`   🗝️  ${row.count} vault items`)
    })

    db.get('SELECT COUNT(*) as count FROM knowledge_graph', (err, row) => {
      if (!err) console.log(`   🧠 ${row.count} knowledge connections`)
    })

    console.log(
      '\n🚀 Your Neural Odyssey awaits! Run `npm run dev` to start learning.'
    )

    // Close database
    db.close(err => {
      if (err) {
        console.error('❌ Error closing database:', err.message)
      } else {
        console.log('✅ Database connection closed')
      }
    })
  } catch (error) {
    console.error('❌ Database initialization failed:', error.message)
    process.exit(1)
  }
}

// Run initialization
initializeDatabase()
