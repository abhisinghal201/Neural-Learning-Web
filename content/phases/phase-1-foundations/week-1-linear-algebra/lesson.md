# Week 1: Linear Algebra Through the Lens of Data

## Overview

Welcome to your Neural Odyssey! This first week introduces you to linear algebra—not as abstract mathematics, but as the fundamental language of machine learning. Every neural network, every data transformation, every optimization algorithm speaks in the language of vectors and matrices.

**Why Linear Algebra Matters for ML:**
- Neural networks are just matrix operations chained together
- Data preprocessing relies heavily on linear transformations
- Principal Component Analysis (PCA) is pure linear algebra
- Gradient descent navigates through multi-dimensional spaces
- Even simple linear regression is solving a linear algebra problem

## Learning Objectives

By the end of this week, you will:

1. **Understand vectors and matrices as data structures** - See how data naturally fits into these mathematical objects
2. **Master fundamental operations** - Addition, multiplication, and transformations that power ML algorithms
3. **Visualize geometric interpretations** - Understand what these operations actually do to data
4. **Implement from scratch** - Build matrix operations without NumPy to understand the mechanics
5. **Connect to real applications** - See how major tech companies use these concepts

## Daily Structure

### Day 1: Vectors as Data Points
**Morning Theory (25 min):**
- What is a vector? (Not just an arrow!)
- Vectors as feature representations
- Vector spaces and dimensions

**Afternoon Coding (25 min):**
- Implement vector operations in pure Python
- Visualize vectors as data points
- Practice with real datasets

### Day 2: Matrices as Data Transformations
**Morning Theory (25 min):**
- Matrices as collections of vectors
- Matrix-vector multiplication as data transformation
- Understanding the shape and meaning

**Afternoon Coding (25 min):**
- Build matrix multiplication from scratch
- Apply transformations to data
- Visualize before/after transformations

### Day 3: Systems of Equations and Data Fitting
**Morning Theory (25 min):**
- Linear systems in ML context
- Overdetermined systems (more data than unknowns)
- Least squares solution

**Afternoon Coding (25 min):**
- Solve linear regression with matrix operations
- Implement normal equation
- Compare with sklearn results

### Day 4: Geometric Transformations
**Morning Theory (25 min):**
- Rotation, scaling, reflection matrices
- Change of basis
- Coordinate systems in data

**Afternoon Coding (25 min):**
- Implement 2D/3D transformations
- Visualize data rotations
- PCA preview (rotation to principal axes)

## Core Concepts

### 1. Vectors: The Building Blocks

A vector isn't just "magnitude and direction"—in ML, it's a data point living in feature space.

```python
# A person's features as a vector
person = [age, height, weight, income, education_years]
# This is a 5-dimensional vector in "person space"
```

**Key Operations:**
- **Addition**: Combining features or moving in feature space
- **Scalar multiplication**: Scaling all features proportionally  
- **Dot product**: Measuring similarity between data points
- **Norm**: Distance from origin (or data magnitude)

### 2. Matrices: Data Collections and Transformations

Matrices serve two crucial roles in ML:

**Role 1: Data Storage**
```python
# Each row is a data point, each column is a feature
data_matrix = [
    [25, 170, 70, 50000, 16],  # Person 1
    [30, 180, 80, 75000, 18],  # Person 2
    [22, 165, 65, 35000, 14],  # Person 3
]
# Shape: (3 people, 5 features)
```

**Role 2: Transformations**
```python
# A matrix can transform data from one space to another
# Example: Convert RGB to grayscale
rgb_to_gray = [
    [0.299, 0.587, 0.114]  # Weighted combination
]
# Multiply by RGB vector to get grayscale value
```

### 3. Matrix Multiplication: The Heart of ML

Matrix multiplication isn't just a mathematical operation—it's how neural networks think:

```python
# Neural network layer
# input: (batch_size, input_features)
# weights: (input_features, output_features)  
# output: (batch_size, output_features)
output = input @ weights + bias
```

Every forward pass through a neural network is a series of matrix multiplications!

## Real-World Connections

### Google's PageRank: Linear Algebra in Action

The algorithm that made Google billions is fundamentally about finding the dominant eigenvector of a massive matrix. The web is represented as a matrix where entry (i,j) represents a link from page i to page j. PageRank finds the steady-state probability distribution—pure linear algebra!

### Image Processing

Every Instagram filter, every photo enhancement, every computer vision algorithm starts with representing images as matrices and applying linear transformations:

```python
# Image as matrix
image = [
    [pixel_1_1, pixel_1_2, ...],
    [pixel_2_1, pixel_2_2, ...],
    ...
]

# Apply filter (convolution is matrix multiplication)
filtered_image = convolution_matrix @ image
```

## Hands-On Projects

### Project 1: Build a Mini-NumPy
Create your own matrix library with basic operations:
- Vector addition and scalar multiplication
- Matrix multiplication (implement the triple loop!)
- Matrix transpose
- Basic linear system solver

### Project 2: Data Transformation Visualizer
Build an interactive tool that shows how matrix transformations affect 2D data:
- Rotation matrices
- Scaling matrices
- Reflection matrices
- Shear transformations

### Project 3: Linear Regression from Scratch
Implement linear regression using only your matrix operations:
- Set up the normal equation: θ = (X^T X)^(-1) X^T y
- Compare results with sklearn
- Visualize the geometric interpretation

## Common Pitfalls and Insights

### Pitfall 1: "Matrices are just number grids"
**Reality**: Matrices represent relationships and transformations. Each multiplication tells a story about how data moves through space.

### Pitfall 2: "Matrix multiplication is complicated"
**Reality**: It's just computing dot products systematically. Each entry in the result is a dot product of a row and column.

### Pitfall 3: "This is too abstract for real ML"
**Reality**: Every time you call `model.fit()` or `neural_net.forward()`, you're using these exact concepts!

## Historical Context

Linear algebra as we know it emerged from trying to solve systems of equations systematically. The key insights:

- **Gauss (1800s)**: Systematic elimination methods
- **Hamilton (1843)**: Vector algebra and quaternions  
- **Sylvester (1850)**: Coined the term "matrix"
- **Cayley (1858)**: Matrix algebra as we know it

These mathematical tools, developed centuries ago, became the foundation of modern AI. Sometimes the most powerful innovations come from rediscovering old mathematics!

## What's Next?

Week 2 will dive deeper into eigenvalues and eigenvectors—the mathematical concepts that unlock Principal Component Analysis and help us understand the "natural directions" in our data. We'll see how Google's PageRank algorithm is really just finding the most important eigenvector of the web!

## Additional Resources

**For Visual Learners:**
- 3Blue1Brown's "Essence of Linear Algebra" series
- Interactive matrix transformation visualizer

**For Deep Divers:**
- Gilbert Strang's MIT Linear Algebra course
- "Linear Algebra and Its Applications" by David Lay

**For ML Connections:**
- "Mathematics for Machine Learning" by Deisenroth, Faisal, and Ong
- Fast.ai's computational linear algebra course

Remember: Every expert was once a beginner. Linear algebra might seem abstract now, but by the end of this week, you'll see matrices and vectors everywhere in the world of data and AI!