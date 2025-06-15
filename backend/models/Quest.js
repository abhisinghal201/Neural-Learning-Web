/**
 * Neural Odyssey Quest Model
 * 
 * Defines quest structures, unlock logic, and quest generation for the ML learning path.
 * Quests are hands-on coding challenges that reinforce theoretical learning with practical implementation.
 * 
 * Quest Types:
 * - coding_exercise: Small coding challenges (15-45 min)
 * - implementation_project: Larger projects (1-3 hours) 
 * - theory_quiz: Interactive concept validation
 * - practical_application: Real-world problem solving
 * 
 * Features:
 * - Dynamic quest generation based on learning progress
 * - Difficulty progression and adaptive challenges
 * - Code solution storage and execution tracking
 * - Hint system and mentoring integration
 * - Achievement and skill point calculations
 * 
 * Connects to: backend/controllers/learningController.js + backend/config/db.js
 * Uses: quest_completions table + learning_progress table
 * 
 * Author: Neural Explorer
 */

const db = require('../config/db');

class Quest {
    constructor(data = {}) {
        this.id = data.id || this.generateQuestId();
        this.title = data.title || '';
        this.description = data.description || '';
        this.type = data.type || 'coding_exercise'; // coding_exercise, implementation_project, theory_quiz, practical_application
        this.phase = data.phase || 1;
        this.week = data.week || 1;
        this.difficulty_level = data.difficulty_level || 1; // 1-5 scale
        this.estimated_time_minutes = data.estimated_time_minutes || 30;
        this.prerequisites = data.prerequisites || []; // Array of lesson IDs
        this.learning_objectives = data.learning_objectives || [];
        this.starter_code = data.starter_code || '';
        this.solution_template = data.solution_template || '';
        this.test_cases = data.test_cases || [];
        this.hints = data.hints || [];
        this.resources = data.resources || [];
        this.tags = data.tags || [];
        this.created_at = data.created_at || new Date().toISOString();
    }

    generateQuestId() {
        return `quest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // ==========================================
    // QUEST DEFINITIONS BY PHASE/WEEK
    // ==========================================

    static getQuestDefinitions() {
        return {
            // Phase 1: Mathematical Foundations (Weeks 1-12)
            phase1: {
                week1: [
                    {
                        id: 'quest_p1w1_matrix_ops',
                        title: 'Build a Matrix Operations Library',
                        description: 'Implement matrix addition, multiplication, and transpose from scratch using only Python lists. No NumPy allowed!',
                        type: 'coding_exercise',
                        phase: 1,
                        week: 1,
                        difficulty_level: 2,
                        estimated_time_minutes: 45,
                        learning_objectives: [
                            'Understand matrix operations at a fundamental level',
                            'Practice algorithmic thinking',
                            'Build intuition for linear algebra computations'
                        ],
                        starter_code: `# Matrix Operations Library
# Implement the following functions using only Python lists

def matrix_add(A, B):
    """Add two matrices A and B"""
    # Your implementation here
    pass

def matrix_multiply(A, B):
    """Multiply two matrices A and B"""
    # Your implementation here
    pass

def matrix_transpose(A):
    """Transpose matrix A"""
    # Your implementation here
    pass

# Test your implementation
if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    
    print("A + B =", matrix_add(A, B))
    print("A * B =", matrix_multiply(A, B))
    print("A^T =", matrix_transpose(A))`,
                        test_cases: [
                            {
                                input: { A: [[1, 2], [3, 4]], B: [[5, 6], [7, 8]] },
                                expected_add: [[6, 8], [10, 12]],
                                expected_multiply: [[19, 22], [43, 50]],
                                expected_transpose_A: [[1, 3], [2, 4]]
                            }
                        ],
                        hints: [
                            "For matrix addition, add corresponding elements: C[i][j] = A[i][j] + B[i][j]",
                            "For matrix multiplication, use the dot product: C[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))",
                            "For transpose, swap rows and columns: C[j][i] = A[i][j]"
                        ],
                        tags: ['linear-algebra', 'matrix-operations', 'algorithms']
                    },
                    {
                        id: 'quest_p1w1_vector_geometry',
                        title: 'Geometric Vector Visualizer',
                        description: 'Create a program that visualizes vector operations (addition, scaling, dot product) and their geometric interpretations.',
                        type: 'implementation_project',
                        phase: 1,
                        week: 1,
                        difficulty_level: 3,
                        estimated_time_minutes: 90,
                        learning_objectives: [
                            'Connect algebraic vector operations with geometric intuition',
                            'Visualize mathematical concepts',
                            'Build foundation for understanding ML geometrically'
                        ],
                        starter_code: `import matplotlib.pyplot as plt
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        # Implement vector addition
        pass
    
    def __mul__(self, scalar):
        # Implement scalar multiplication
        pass
    
    def dot(self, other):
        # Implement dot product
        pass
    
    def magnitude(self):
        # Implement magnitude calculation
        pass
    
    def plot(self, ax, color='blue', label=None):
        # Plot the vector as an arrow from origin
        ax.arrow(0, 0, self.x, self.y, head_width=0.1, head_length=0.1, 
                fc=color, ec=color, label=label)

# Create visualization demo
def demo_vector_operations():
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create vectors
    v1 = Vector(3, 2)
    v2 = Vector(1, 3)
    
    # Visualize operations
    # Your implementation here
    
    plt.show()

if __name__ == "__main__":
    demo_vector_operations()`,
                        hints: [
                            "Use matplotlib.pyplot to create vector visualizations",
                            "Vector addition can be shown as a parallelogram",
                            "Dot product relates to the angle between vectors: v1 • v2 = |v1||v2|cos(θ)"
                        ],
                        tags: ['vectors', 'visualization', 'geometry', 'matplotlib']
                    }
                ],
                week2: [
                    {
                        id: 'quest_p1w2_eigenvalue_finder',
                        title: 'Eigenvalue Detective',
                        description: 'Implement the power iteration method to find the dominant eigenvalue and eigenvector of a matrix. Understand why this is the foundation of PageRank!',
                        type: 'coding_exercise',
                        phase: 1,
                        week: 2,
                        difficulty_level: 4,
                        estimated_time_minutes: 60,
                        learning_objectives: [
                            'Understand eigenvalues and eigenvectors computationally',
                            'Implement iterative numerical methods',
                            'Connect to real-world applications like PageRank'
                        ],
                        starter_code: `import numpy as np

def power_iteration(A, num_iterations=100, tolerance=1e-6):
    """
    Find the dominant eigenvalue and eigenvector using power iteration
    
    Args:
        A: Square matrix (as nested list or numpy array)
        num_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        eigenvalue: Dominant eigenvalue
        eigenvector: Corresponding eigenvector
        converged: Whether the method converged
    """
    # Convert to numpy array if needed
    A = np.array(A)
    n = A.shape[0]
    
    # Start with random vector
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize
    
    # Your implementation here
    pass

def demonstrate_pagerank_connection():
    """
    Show how eigenvalues relate to PageRank algorithm
    """
    # Create a simple web graph transition matrix
    # Web pages: A -> B, B -> C, C -> A (simple cycle)
    web_graph = [
        [0, 1, 0],  # Page A links to B
        [0, 0, 1],  # Page B links to C  
        [1, 0, 0]   # Page C links to A
    ]
    
    print("Web Graph Transition Matrix:")
    print(np.array(web_graph))
    
    # Find dominant eigenvector (PageRank scores)
    eigenval, eigenvec, converged = power_iteration(web_graph)
    
    print(f"\\nPageRank Scores (eigenvector): {eigenvec}")
    print(f"Eigenvalue: {eigenval}")
    print(f"Converged: {converged}")

if __name__ == "__main__":
    demonstrate_pagerank_connection()`,
                        test_cases: [
                            {
                                input: [[2, 1], [1, 2]],
                                expected_eigenvalue_approx: 3.0,
                                expected_eigenvector_approx: [0.707, 0.707]
                            }
                        ],
                        hints: [
                            "Power iteration: repeatedly multiply by matrix and normalize",
                            "The dominant eigenvalue is the one with largest absolute value", 
                            "Check convergence by comparing consecutive iterations",
                            "PageRank is literally finding the principal eigenvector of the web graph!"
                        ],
                        tags: ['eigenvalues', 'eigenvectors', 'pagerank', 'numerical-methods']
                    }
                ],
                week3: [
                    {
                        id: 'quest_p1w3_gradient_descent',
                        title: 'Build Gradient Descent from Scratch',
                        description: 'Implement gradient descent to find the minimum of a function. Watch the algorithm "learn" by following the slope downhill!',
                        type: 'coding_exercise',
                        phase: 1,
                        week: 3,
                        difficulty_level: 3,
                        estimated_time_minutes: 50,
                        learning_objectives: [
                            'Understand gradient descent algorithm fundamentally',
                            'Implement automatic differentiation basics',
                            'Visualize optimization process'
                        ],
                        starter_code: `import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x_start, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Implement gradient descent optimization
    
    Args:
        f: Function to minimize
        grad_f: Gradient function (derivative of f)
        x_start: Starting point
        learning_rate: Step size
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        x_history: List of x values during optimization
        f_history: List of function values during optimization
        converged: Whether algorithm converged
    """
    x = x_start
    x_history = [x]
    f_history = [f(x)]
    
    # Your implementation here
    pass

# Test functions
def quadratic_function(x):
    """Simple quadratic: f(x) = (x-3)^2 + 1"""
    return (x - 3)**2 + 1

def quadratic_gradient(x):
    """Gradient of quadratic: f'(x) = 2(x-3)"""
    return 2 * (x - 3)

def rosenbrock_2d(xy):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient_2d(xy):
    """Gradient of Rosenbrock function"""
    x, y = xy
    grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

def visualize_optimization():
    """Create visualizations of the optimization process"""
    # 1D example
    x_history, f_history, converged = gradient_descent(
        quadratic_function, quadratic_gradient, x_start=0
    )
    
    # Plot optimization path
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x_range = np.linspace(-2, 8, 100)
    y_range = [quadratic_function(x) for x in x_range]
    plt.plot(x_range, y_range, 'b-', label='f(x) = (x-3)² + 1')
    plt.plot(x_history, f_history, 'ro-', alpha=0.7, label='Gradient Descent Path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('1D Gradient Descent')
    plt.legend()
    plt.grid(True)
    
    # Your 2D visualization code here
    plt.subplot(1, 2, 2)
    # Implement 2D Rosenbrock visualization
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_optimization()`,
                        hints: [
                            "Update rule: x_new = x_old - learning_rate * gradient",
                            "Check convergence when gradient magnitude is small",
                            "For 2D visualization, use contour plots to show the function landscape",
                            "Try different learning rates to see their effect on convergence"
                        ],
                        tags: ['optimization', 'gradient-descent', 'calculus', 'visualization']
                    }
                ],
                week4: [
                    {
                        id: 'quest_p1w4_backprop_neural_net',
                        title: 'Neural Network with Backpropagation',
                        description: 'Build a simple neural network from scratch and implement backpropagation. Watch the network learn to solve XOR - the problem that caused the first AI winter!',
                        type: 'implementation_project',
                        phase: 1,
                        week: 4,
                        difficulty_level: 5,
                        estimated_time_minutes: 120,
                        learning_objectives: [
                            'Understand backpropagation algorithm deeply',
                            'Implement neural networks from first principles',
                            'Solve the historic XOR problem'
                        ],
                        starter_code: `import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers):
        """
        Initialize neural network
        layers: list of integers representing number of neurons in each layer
        """
        self.layers = layers
        self.num_layers = len(layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Your initialization code here
        pass
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        X: input data
        Returns: output of network and intermediate values for backprop
        """
        # Your implementation here
        pass
    
    def backward_propagation(self, X, y, cache):
        """
        Backward propagation to compute gradients
        X: input data
        y: true labels
        cache: stored values from forward propagation
        Returns: gradients for weights and biases
        """
        # Your implementation here
        pass
    
    def update_parameters(self, gradients, learning_rate):
        """Update weights and biases using gradients"""
        # Your implementation here
        pass
    
    def train(self, X, y, epochs, learning_rate):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            output, cache = self.forward_propagation(X)
            
            # Compute loss
            loss = np.mean((y - output)**2)
            losses.append(loss)
            
            # Backward propagation
            gradients = self.backward_propagation(X, y, cache)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        output, _ = self.forward_propagation(X)
        return output

def create_xor_dataset():
    """Create XOR dataset - the problem that stumped perceptrons"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR output
    return X, y

def demonstrate_xor_learning():
    """Demonstrate neural network learning XOR"""
    # Create XOR dataset
    X, y = create_xor_dataset()
    
    print("XOR Dataset:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}")
    
    # Create and train neural network
    # Architecture: 2 inputs -> 4 hidden -> 1 output
    nn = NeuralNetwork([2, 4, 1])
    
    print("\\nTraining neural network to solve XOR...")
    losses = nn.train(X, y, epochs=1000, learning_rate=1.0)
    
    # Test the trained network
    print("\\nTesting trained network:")
    predictions = nn.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]:.0f}, Prediction: {predictions[i][0]:.3f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Visualize decision boundary
    plt.subplot(1, 2, 2)
    visualize_decision_boundary(nn)
    plt.title('XOR Decision Boundary')
    
    plt.tight_layout()
    plt.show()

def visualize_decision_boundary(nn):
    """Visualize the decision boundary learned by the network"""
    # Create a mesh of points
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar()
    
    # Plot XOR points
    X, y = create_xor_dataset()
    colors = ['red', 'blue']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], c=colors[int(y[i][0])], 
                   s=100, alpha=0.9, edgecolors='black')

if __name__ == "__main__":
    demonstrate_xor_learning()`,
                        hints: [
                            "Initialize weights with small random values (e.g., Gaussian with std=0.5)",
                            "Use sigmoid activation function: σ(z) = 1/(1+e^(-z))",
                            "Forward prop: compute activations layer by layer",
                            "Backward prop: use chain rule to compute gradients layer by layer",
                            "The XOR problem requires a hidden layer - that's why it defeated the perceptron!"
                        ],
                        tags: ['neural-networks', 'backpropagation', 'xor-problem', 'deep-learning']
                    }
                ]
            },

            // Phase 2: Core Machine Learning (Weeks 13-24)
            phase2: {
                week13: [
                    {
                        id: 'quest_p2w13_ml_pipeline',
                        title: 'End-to-End ML Pipeline',
                        description: 'Build a complete machine learning pipeline from data loading to model deployment. Experience the full ML workflow!',
                        type: 'implementation_project',
                        phase: 2,
                        week: 13,
                        difficulty_level: 4,
                        estimated_time_minutes: 90,
                        learning_objectives: [
                            'Understand complete ML workflow',
                            'Practice data preprocessing and feature engineering',
                            'Implement model evaluation and selection'
                        ],
                        starter_code: `# Complete ML Pipeline Implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
    
    def load_and_explore_data(self, data_path):
        """Load data and perform initial exploration"""
        # Your implementation here
        pass
    
    def preprocess_data(self, df, target_column):
        """Clean and preprocess the data"""
        # Your implementation here
        pass
    
    def train_models(self, X_train, y_train):
        """Train multiple models and compare performance"""
        # Your implementation here
        pass
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        # Your implementation here
        pass
    
    def deploy_best_model(self):
        """Prepare the best model for deployment"""
        # Your implementation here
        pass

# Example usage and testing
if __name__ == "__main__":
    pipeline = MLPipeline()
    
    # Create sample dataset (or load real data)
    # Implement your complete pipeline here`,
                        tags: ['ml-pipeline', 'data-preprocessing', 'model-evaluation']
                    }
                ]
            },

            // Phase 3: Advanced Topics (Weeks 25-36)
            phase3: {
                week25: [
                    {
                        id: 'quest_p3w25_transformer_attention',
                        title: 'Build Transformer Attention from Scratch',
                        description: 'Implement the attention mechanism that revolutionized AI. Understand why "Attention is All You Need"!',
                        type: 'implementation_project',
                        phase: 3,
                        week: 25,
                        difficulty_level: 5,
                        estimated_time_minutes: 150,
                        learning_objectives: [
                            'Understand attention mechanism deeply',
                            'Implement multi-head attention',
                            'Build foundation for modern LLMs'
                        ],
                        starter_code: `# Transformer Attention Implementation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Your implementation here
        pass
    
    def forward(self, query, key, value, mask=None):
        # Your implementation here
        pass

# Complete transformer implementation and testing`,
                        tags: ['transformers', 'attention', 'modern-ai']
                    }
                ]
            },

            // Phase 4: Mastery (Weeks 37-48)
            phase4: {
                week37: [
                    {
                        id: 'quest_p4w37_research_project',
                        title: 'Original Research Project',
                        description: 'Design and execute an original ML research project. Contribute something new to the field!',
                        type: 'practical_application',
                        phase: 4,
                        week: 37,
                        difficulty_level: 5,
                        estimated_time_minutes: 300,
                        learning_objectives: [
                            'Conduct original research',
                            'Apply advanced ML techniques',
                            'Communicate findings effectively'
                        ],
                        starter_code: `# Research Project Template
# Choose your research direction and implement novel solutions`,
                        tags: ['research', 'innovation', 'mastery']
                    }
                ]
            }
        };
    }

    // ==========================================
    // QUEST RETRIEVAL AND GENERATION
    // ==========================================

    static async getQuestsForWeek(phase, week) {
        try {
            const definitions = Quest.getQuestDefinitions();
            const phaseKey = `phase${phase}`;
            const weekKey = `week${week}`;
            
            if (definitions[phaseKey] && definitions[phaseKey][weekKey]) {
                return definitions[phaseKey][weekKey].map(questData => new Quest(questData));
            }
            
            // Generate adaptive quests if no predefined ones exist
            return Quest.generateAdaptiveQuests(phase, week);
            
        } catch (error) {
            console.error('Error getting quests for week:', error);
            return [];
        }
    }

    static async generateAdaptiveQuests(phase, week) {
        // Generate quests based on user progress and current learning phase
        const baseQuests = [
            {
                id: `quest_p${phase}w${week}_adaptive_coding`,
                title: `Phase ${phase} Week ${week} Coding Challenge`,
                description: `Apply the concepts from Phase ${phase}, Week ${week} in a practical coding exercise.`,
                type: 'coding_exercise',
                phase,
                week,
                difficulty_level: Math.min(5, Math.max(1, phase)),
                estimated_time_minutes: 30 + (phase * 15),
                starter_code: `# Phase ${phase} Week ${week} Challenge
# Implement the concepts you've learned this week

def main():
    # Your implementation here
    pass

if __name__ == "__main__":
    main()`,
                tags: [`phase-${phase}`, `week-${week}`, 'adaptive']
            }
        ];
        
        return baseQuests.map(questData => new Quest(questData));
    }

    static async getAvailableQuests(userPhase, userWeek) {
        try {
            const availableQuests = [];
            
            // Get quests for current and previous weeks
            for (let phase = 1; phase <= userPhase; phase++) {
                const maxWeek = phase === userPhase ? userWeek : 12; // Assuming 12 weeks per phase
                
                for (let week = 1; week <= maxWeek; week++) {
                    const weekQuests = await Quest.getQuestsForWeek(phase, week);
                    availableQuests.push(...weekQuests);
                }
            }
            
            return availableQuests;
            
        } catch (error) {
            console.error('Error getting available quests:', error);
            return [];
        }
    }

    // ==========================================
    // QUEST COMPLETION TRACKING
    // ==========================================

    async save() {
        try {
            const result = await db.run(`
                INSERT INTO quest_completions (
                    quest_id, quest_title, quest_type, phase, week, difficulty_level,
                    status, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(quest_id) DO UPDATE SET
                    quest_title = excluded.quest_title,
                    quest_type = excluded.quest_type,
                    difficulty_level = excluded.difficulty_level,
                    status = excluded.status,
                    completed_at = excluded.completed_at
            `, [
                this.id,
                this.title,
                this.type,
                this.phase,
                this.week,
                this.difficulty_level,
                'available',
                new Date().toISOString()
            ]);
            
            return result;
            
        } catch (error) {
            console.error('Error saving quest:', error);
            throw error;
        }
    }

    static async getCompletedQuests(userId = 1) {
        try {
            const completedQuests = await db.query(`
                SELECT * FROM quest_completions 
                WHERE status IN ('completed', 'mastered')
                ORDER BY completed_at DESC
            `);
            
            return completedQuests;
            
        } catch (error) {
            console.error('Error getting completed quests:', error);
            return [];
        }
    }

    static async getQuestProgress(phase = null, week = null) {
        try {
            let whereClause = '1=1';
            const params = [];
            
            if (phase !== null) {
                whereClause += ' AND phase = ?';
                params.push(phase);
            }
            
            if (week !== null) {
                whereClause += ' AND week = ?';
                params.push(week);
            }
            
            const progress = await db.query(`
                SELECT 
                    phase,
                    week,
                    COUNT(*) as total_quests,
                    COUNT(CASE WHEN status = 'completed' OR status = 'mastered' THEN 1 END) as completed_quests,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_quests,
                    AVG(difficulty_level) as avg_difficulty,
                    SUM(time_to_complete_minutes) as total_time_spent
                FROM quest_completions 
                WHERE ${whereClause}
                GROUP BY phase, week
                ORDER BY phase, week
            `, params);
            
            return progress;
            
        } catch (error) {
            console.error('Error getting quest progress:', error);
            return [];
        }
    }

    // ==========================================
    // QUEST DIFFICULTY AND ADAPTATION
    // ==========================================

    static calculateDynamicDifficulty(userStats) {
        // Adaptive difficulty based on user performance
        const {
            completion_rate = 0.7,
            average_attempts = 2,
            average_time_ratio = 1.0, // actual_time / estimated_time
            mastery_rate = 0.3
        } = userStats;

        let difficulty_modifier = 0;

        // Adjust based on completion rate
        if (completion_rate > 0.9) difficulty_modifier += 0.5;
        else if (completion_rate < 0.5) difficulty_modifier -= 0.5;

        // Adjust based on attempts needed
        if (average_attempts < 1.5) difficulty_modifier += 0.3;
        else if (average_attempts > 3) difficulty_modifier -= 0.3;

        // Adjust based on time taken
        if (average_time_ratio < 0.7) difficulty_modifier += 0.2;
        else if (average_time_ratio > 1.5) difficulty_modifier -= 0.2;

        // Adjust based on mastery rate
        if (mastery_rate > 0.6) difficulty_modifier += 0.4;
        else if (mastery_rate < 0.2) difficulty_modifier -= 0.4;

        return Math.max(1, Math.min(5, 3 + difficulty_modifier));
    }

    static async getUserStats(userId = 1) {
        try {
            const stats = await db.get(`
                SELECT 
                    COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) * 1.0 / COUNT(*) as completion_rate,
                    AVG(attempts_count) as average_attempts,
                    AVG(time_to_complete_minutes * 1.0 / NULLIF(estimated_time_minutes, 0)) as average_time_ratio,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) * 1.0 / COUNT(*) as mastery_rate
                FROM quest_completions qc
                JOIN (
                    SELECT quest_id, estimated_time_minutes 
                    FROM quest_definitions 
                ) qd ON qc.quest_id = qd.quest_id
                WHERE qc.completed_at >= date('now', '-30 days')
            `);
            
            return stats || {
                completion_rate: 0.7,
                average_attempts: 2,
                average_time_ratio: 1.0,
                mastery_rate: 0.3
            };
            
        } catch (error) {
            console.error('Error getting user stats:', error);
            return {
                completion_rate: 0.7,
                average_attempts: 2,
                average_time_ratio: 1.0,
                mastery_rate: 0.3
            };
        }
    }

    // ==========================================
    // QUEST VALIDATION AND TESTING
    // ==========================================

    static validateQuestSolution(questId, userCode, testCases) {
        try {
            const results = {
                passed: 0,
                total: testCases.length,
                test_results: [],
                execution_error: null,
                performance_score: 0
            };

            // This would implement code execution and testing
            // For security, this should be in a sandboxed environment
            console.log(`Validating solution for quest ${questId}`);
            console.log('User code:', userCode);
            
            // Placeholder validation logic
            // In production, this would execute user code safely and run test cases
            testCases.forEach((testCase, index) => {
                const testResult = {
                    test_id: index,
                    passed: Math.random() > 0.3, // Placeholder
                    expected: testCase.expected || 'Expected output',
                    actual: 'User output', // Would be actual execution result
                    execution_time: Math.random() * 100 // ms
                };
                
                results.test_results.push(testResult);
                if (testResult.passed) results.passed++;
            });

            results.performance_score = (results.passed / results.total) * 100;
            
            return results;
            
        } catch (error) {
            return {
                passed: 0,
                total: testCases.length,
                test_results: [],
                execution_error: error.message,
                performance_score: 0
            };
        }
    }

    // ==========================================
    // QUEST RECOMMENDATIONS
    // ==========================================

    static async getRecommendedQuests(userId = 1, limit = 5) {
        try {
            // Get user's current progress
            const userProfile = await db.get(`
                SELECT current_phase, current_week FROM user_profile WHERE id = ?
            `, [userId]);

            if (!userProfile) {
                return [];
            }

            // Get user performance stats
            const userStats = await Quest.getUserStats(userId);
            const recommendedDifficulty = Quest.calculateDynamicDifficulty(userStats);

            // Get completed quest IDs
            const completedQuests = await db.query(`
                SELECT quest_id FROM quest_completions 
                WHERE status IN ('completed', 'mastered')
            `);
            const completedIds = new Set(completedQuests.map(q => q.quest_id));

            // Get available quests for current and nearby weeks
            const availableQuests = await Quest.getAvailableQuests(
                userProfile.current_phase, 
                userProfile.current_week
            );

            // Filter and score quests
            const recommendations = availableQuests
                .filter(quest => !completedIds.has(quest.id))
                .map(quest => {
                    let score = 100; // Base score

                    // Prefer quests close to user's current week
                    const weekDistance = Math.abs(
                        (quest.phase - userProfile.current_phase) * 12 + 
                        (quest.week - userProfile.current_week)
                    );
                    score -= weekDistance * 10;

                    // Prefer quests matching recommended difficulty
                    const difficultyMatch = Math.abs(quest.difficulty_level - recommendedDifficulty);
                    score -= difficultyMatch * 15;

                    // Boost coding exercises if user is doing well
                    if (quest.type === 'coding_exercise' && userStats.completion_rate > 0.8) {
                        score += 20;
                    }

                    // Boost theory quests if user struggles with coding
                    if (quest.type === 'theory_quiz' && userStats.completion_rate < 0.6) {
                        score += 15;
                    }

                    return {
                        ...quest,
                        recommendation_score: Math.max(0, score),
                        difficulty_match: difficultyMatch === 0
                    };
                })
                .sort((a, b) => b.recommendation_score - a.recommendation_score)
                .slice(0, limit);

            return recommendations;
            
        } catch (error) {
            console.error('Error getting recommended quests:', error);
            return [];
        }
    }

    // ==========================================
    // QUEST ANALYTICS
    // ==========================================

    static async getQuestAnalytics(timeframe = '30 days') {
        try {
            const analytics = await db.get(`
                SELECT 
                    COUNT(*) as total_attempts,
                    COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as successful_completions,
                    AVG(attempts_count) as avg_attempts_per_quest,
                    AVG(time_to_complete_minutes) as avg_completion_time,
                    AVG(difficulty_level) as avg_difficulty,
                    COUNT(DISTINCT quest_type) as quest_types_attempted
                FROM quest_completions 
                WHERE completed_at >= date('now', '-${timeframe.split(' ')[0]} days')
            `);

            const typeBreakdown = await db.query(`
                SELECT 
                    quest_type,
                    COUNT(*) as attempts,
                    COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completions,
                    AVG(time_to_complete_minutes) as avg_time
                FROM quest_completions 
                WHERE completed_at >= date('now', '-${timeframe.split(' ')[0]} days')
                GROUP BY quest_type
                ORDER BY attempts DESC
            `);

            const difficultyBreakdown = await db.query(`
                SELECT 
                    difficulty_level,
                    COUNT(*) as attempts,
                    COUNT(CASE WHEN status IN ('completed', 'mastered') THEN 1 END) as completions,
                    AVG(attempts_count) as avg_attempts
                FROM quest_completions 
                WHERE completed_at >= date('now', '-${timeframe.split(' ')[0]} days')
                GROUP BY difficulty_level
                ORDER BY difficulty_level
            `);

            return {
                overview: analytics,
                by_type: typeBreakdown,
                by_difficulty: difficultyBreakdown,
                timeframe,
                generated_at: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('Error getting quest analytics:', error);
            return null;
        }
    }

    // ==========================================
    // UTILITY METHODS
    // ==========================================

    toJSON() {
        return {
            id: this.id,
            title: this.title,
            description: this.description,
            type: this.type,
            phase: this.phase,
            week: this.week,
            difficulty_level: this.difficulty_level,
            estimated_time_minutes: this.estimated_time_minutes,
            prerequisites: this.prerequisites,
            learning_objectives: this.learning_objectives,
            starter_code: this.starter_code,
            solution_template: this.solution_template,
            test_cases: this.test_cases,
            hints: this.hints,
            resources: this.resources,
            tags: this.tags,
            created_at: this.created_at
        };
    }

    static async seedQuestDatabase() {
        /**
         * Populate database with predefined quests
         * Called during initialization
         */
        try {
            const questDefinitions = Quest.getQuestDefinitions();
            let totalSeeded = 0;

            for (const [phaseKey, phaseData] of Object.entries(questDefinitions)) {
                const phase = parseInt(phaseKey.replace('phase', ''));
                
                for (const [weekKey, weekQuests] of Object.entries(phaseData)) {
                    const week = parseInt(weekKey.replace('week', ''));
                    
                    for (const questData of weekQuests) {
                        const quest = new Quest(questData);
                        await quest.save();
                        totalSeeded++;
                    }
                }
            }

            console.log(`✅ Seeded ${totalSeeded} quests to database`);
            return totalSeeded;
            
        } catch (error) {
            console.error('❌ Error seeding quest database:', error);
            throw error;
        }
    }
}

module.exports = Quest;