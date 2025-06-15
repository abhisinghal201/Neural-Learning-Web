"""
Neural Odyssey - Week 14: Linear Models Deep Dive
Phase 2: Core Machine Learning (Week 2)

Mastering the Art of Regularized Linear Learning

Building on Week 13's supervised learning foundations, this week dives deep into the 
world of linear models. You'll explore Ridge, Lasso, and Elastic Net regression,
understand the mathematics of regularization, implement feature selection algorithms,
and master the art of finding the optimal balance between model complexity and performance.

This comprehensive exploration covers:
1. Linear regression revisited with regularization theory
2. Ridge regression: L2 regularization and geometric interpretation
3. Lasso regression: L1 regularization and automatic feature selection
4. Elastic Net: Combining the best of Ridge and Lasso
5. Coordinate descent optimization
6. Regularization paths and cross-validation
7. Feature selection techniques and their trade-offs
8. Generalized linear models (GLMs)

To get started, run: python exercises.py

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.optimize import minimize
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)


# ==========================================
# LINEAR REGRESSION MATHEMATICAL FOUNDATION
# ==========================================

class LinearRegressionFromScratch:
    """
    Linear regression implementation from mathematical first principles
    Demonstrates normal equation, gradient descent, and geometric interpretation
    """
    
    def __init__(self, method='normal_equation'):
        self.method = method
        self.weights = None
        self.bias = None
        self.training_history = {'mse': [], 'weights': [], 'gradients': []}
        
    def fit(self, X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Fit linear regression using specified method
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.method == 'normal_equation':
            return self._fit_normal_equation(X, y)
        elif self.method == 'gradient_descent':
            return self._fit_gradient_descent(X, y, learning_rate, max_iterations, tolerance)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
    
    def _fit_normal_equation(self, X, y):
        """
        Solve using normal equation: θ = (X^T X)^(-1) X^T y
        """
        print(f"   🧮 Solving using Normal Equation")
        print(f"   Formula: θ = (X^T X)^(-1) X^T y")
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Compute normal equation
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        
        print(f"   Matrix X^T X shape: {XtX.shape}")
        print(f"   Condition number: {np.linalg.cond(XtX):.2f}")
        
        if np.linalg.cond(XtX) > 1e12:
            print(f"   ⚠️  Warning: Matrix is ill-conditioned!")
        
        # Solve system
        theta = np.linalg.solve(XtX, Xty)
        self.bias = theta[0]
        self.weights = theta[1:]
        
        # Compute final metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        print(f"   ✅ Converged analytically")
        print(f"   Final MSE: {mse:.6f}")
        print(f"   Final R²: {r2:.6f}")
        
        return self
    
    def _fit_gradient_descent(self, X, y, learning_rate, max_iterations, tolerance):
        """
        Solve using gradient descent optimization
        """
        print(f"   📈 Solving using Gradient Descent")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max iterations: {max_iterations}")
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        self.bias = 0.0
        
        for iteration in range(max_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss and metrics
            mse = np.mean((y - y_pred) ** 2)
            
            # Compute gradients
            residuals = y_pred - y
            grad_weights = (2 / len(X)) * (X.T @ residuals)
            grad_bias = (2 / len(X)) * np.sum(residuals)
            
            # Update parameters
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
            
            # Store history
            self.training_history['mse'].append(mse)
            self.training_history['weights'].append(self.weights.copy())
            self.training_history['gradients'].append(np.linalg.norm(grad_weights))
            
            # Check convergence
            if np.linalg.norm(grad_weights) < tolerance:
                print(f"   ✅ Converged after {iteration + 1} iterations")
                break
                
            if iteration % 200 == 0:
                print(f"   Iteration {iteration}: MSE = {mse:.6f}, ||∇|| = {np.linalg.norm(grad_weights):.6f}")
        
        # Final metrics
        y_pred = self.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        print(f"   Final R²: {r2:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias
    
    def plot_training_history(self):
        """Visualize gradient descent training process"""
        if not self.training_history['mse']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # MSE convergence
        axes[0, 0].plot(self.training_history['mse'], 'b-', linewidth=2)
        axes[0, 0].set_title('MSE Convergence')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Gradient magnitude
        axes[0, 1].plot(self.training_history['gradients'], 'r-', linewidth=2)
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('||∇w||')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Weight evolution
        if len(self.weights) <= 5:  # Only plot if few weights
            weight_history = np.array(self.training_history['weights'])
            for i in range(len(self.weights)):
                axes[1, 0].plot(weight_history[:, i], label=f'w_{i}', linewidth=2)
            axes[1, 0].set_title('Weight Evolution')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Weight Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate analysis
        if len(self.training_history['mse']) > 10:
            window = min(50, len(self.training_history['mse']) // 10)
            smoothed_mse = np.convolve(self.training_history['mse'], 
                                     np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smoothed_mse, 'g-', linewidth=2)
            axes[1, 1].set_title(f'Smoothed MSE (window={window})')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Smoothed MSE')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ==========================================
# RIDGE REGRESSION IMPLEMENTATION
# ==========================================

class RidgeRegressionFromScratch:
    """
    Ridge regression with L2 regularization from mathematical first principles
    Demonstrates shrinkage, bias-variance tradeoff, and geometric interpretation
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Fit Ridge regression using normal equation with L2 penalty
        Minimizes: ||y - Xβ||² + α||β||²
        Solution: β = (X^T X + αI)^(-1) X^T y
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"   🔵 Ridge Regression (α = {self.alpha})")
        print(f"   Objective: ||y - Xβ||² + {self.alpha}||β||²")
        
        # Center data (important for regularization)
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Ridge normal equation: (X^T X + αI)^(-1) X^T y
        XtX = X_centered.T @ X_centered
        identity = np.eye(X_centered.shape[1])
        ridge_matrix = XtX + self.alpha * identity
        Xty = X_centered.T @ y_centered
        
        print(f"   Original condition number: {np.linalg.cond(XtX):.2f}")
        print(f"   Regularized condition number: {np.linalg.cond(ridge_matrix):.2f}")
        
        # Solve regularized system
        self.weights = np.linalg.solve(ridge_matrix, Xty)
        self.bias = y_mean - X_mean @ self.weights
        
        # Store data means for prediction
        self.X_mean = X_mean
        self.y_mean = y_mean
        
        # Compute metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        
        print(f"   MSE: {mse:.6f}, R²: {r2:.6f}")
        print(f"   L2 penalty: {self.alpha * np.sum(self.weights**2):.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_centered = X - self.X_mean
        return X_centered @ self.weights + self.bias
    
    def get_shrinkage_factor(self, X):
        """
        Compute effective degrees of freedom and shrinkage factors
        Shows how regularization shrinks the solution
        """
        X_centered = X - self.X_mean
        XtX = X_centered.T @ X_centered
        identity = np.eye(X_centered.shape[1])
        
        # Hat matrix for Ridge: H = X(X^T X + αI)^(-1)X^T
        ridge_inv = np.linalg.inv(XtX + self.alpha * identity)
        hat_matrix = X_centered @ ridge_inv @ X_centered.T
        
        # Effective degrees of freedom
        df_ridge = np.trace(hat_matrix)
        df_ols = X_centered.shape[1]
        
        print(f"   📊 Shrinkage Analysis:")
        print(f"   OLS degrees of freedom: {df_ols}")
        print(f"   Ridge degrees of freedom: {df_ridge:.2f}")
        print(f"   Shrinkage factor: {1 - df_ridge/df_ols:.3f}")
        
        return df_ridge, hat_matrix


# ==========================================
# LASSO REGRESSION IMPLEMENTATION
# ==========================================

class LassoRegressionFromScratch:
    """
    Lasso regression with L1 regularization using coordinate descent
    Demonstrates automatic feature selection and sparsity
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.feature_path = []
        
    def fit(self, X, y, max_iterations=1000, tolerance=1e-4):
        """
        Fit Lasso regression using coordinate descent algorithm
        Minimizes: ||y - Xβ||² + α||β||₁
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"   🔶 Lasso Regression (α = {self.alpha})")
        print(f"   Objective: ||y - Xβ||² + {self.alpha}||β||₁")
        print(f"   Using coordinate descent optimization")
        
        # Center and standardize
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y)
        
        X_scaled = (X - X_mean) / X_std
        y_centered = y - y_mean
        
        # Initialize weights
        self.weights = np.zeros(X_scaled.shape[1])
        
        for iteration in range(max_iterations):
            weights_old = self.weights.copy()
            
            for j in range(len(self.weights)):
                # Compute partial residual
                partial_residual = y_centered - X_scaled @ self.weights + self.weights[j] * X_scaled[:, j]
                
                # Coordinate-wise update with soft thresholding
                rho = X_scaled[:, j] @ partial_residual
                
                if rho > self.alpha:
                    self.weights[j] = (rho - self.alpha) / (X_scaled[:, j] @ X_scaled[:, j])
                elif rho < -self.alpha:
                    self.weights[j] = (rho + self.alpha) / (X_scaled[:, j] @ X_scaled[:, j])
                else:
                    self.weights[j] = 0.0
            
            # Store feature path
            self.feature_path.append(self.weights.copy())
            
            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < tolerance:
                print(f"   ✅ Converged after {iteration + 1} iterations")
                break
                
            if iteration % 100 == 0:
                active_features = np.sum(np.abs(self.weights) > 1e-8)
                print(f"   Iteration {iteration}: {active_features} active features")
        
        # Transform weights back to original scale
        self.weights = self.weights / X_std
        self.bias = y_mean - X_mean @ self.weights
        
        # Store scaling parameters
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        
        # Final metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        active_features = np.sum(np.abs(self.weights) > 1e-8)
        
        print(f"   MSE: {mse:.6f}, R²: {r2:.6f}")
        print(f"   Active features: {active_features}/{len(self.weights)}")
        print(f"   L1 penalty: {self.alpha * np.sum(np.abs(self.weights)):.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias
    
    def get_feature_importance(self):
        """Return feature importance based on coefficient magnitudes"""
        return np.abs(self.weights)
    
    def plot_regularization_path(self):
        """Visualize how coefficients evolve during coordinate descent"""
        if not self.feature_path:
            print("No regularization path available")
            return
        
        path_array = np.array(self.feature_path)
        
        plt.figure(figsize=(10, 6))
        for j in range(path_array.shape[1]):
            if np.max(np.abs(path_array[:, j])) > 1e-8:  # Only plot non-zero features
                plt.plot(path_array[:, j], label=f'Feature {j}', linewidth=2)
        
        plt.title('Lasso Regularization Path (Coordinate Descent)')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficient Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ==========================================
# ELASTIC NET IMPLEMENTATION
# ==========================================

class ElasticNetFromScratch:
    """
    Elastic Net regression combining L1 and L2 regularization
    Balances feature selection (L1) with coefficient shrinkage (L2)
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # Mix between L1 and L2: 0=Ridge, 1=Lasso
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, max_iterations=1000, tolerance=1e-4):
        """
        Fit Elastic Net using coordinate descent
        Minimizes: ||y - Xβ||² + α(ρ||β||₁ + (1-ρ)/2||β||²)
        where ρ = l1_ratio
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"   🔷 Elastic Net Regression")
        print(f"   α = {self.alpha}, L1 ratio = {self.l1_ratio}")
        print(f"   L1 weight: {self.alpha * self.l1_ratio:.3f}")
        print(f"   L2 weight: {self.alpha * (1 - self.l1_ratio):.3f}")
        
        # Center and standardize
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y)
        
        X_scaled = (X - X_mean) / X_std
        y_centered = y - y_mean
        
        # Initialize weights
        self.weights = np.zeros(X_scaled.shape[1])
        
        # Precompute for efficiency
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)
        
        for iteration in range(max_iterations):
            weights_old = self.weights.copy()
            
            for j in range(len(self.weights)):
                # Compute partial residual
                partial_residual = y_centered - X_scaled @ self.weights + self.weights[j] * X_scaled[:, j]
                
                # Elastic Net coordinate update
                rho = X_scaled[:, j] @ partial_residual
                denominator = X_scaled[:, j] @ X_scaled[:, j] + l2_penalty
                
                if rho > l1_penalty:
                    self.weights[j] = (rho - l1_penalty) / denominator
                elif rho < -l1_penalty:
                    self.weights[j] = (rho + l1_penalty) / denominator
                else:
                    self.weights[j] = 0.0
            
            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < tolerance:
                print(f"   ✅ Converged after {iteration + 1} iterations")
                break
                
            if iteration % 100 == 0:
                active_features = np.sum(np.abs(self.weights) > 1e-8)
                print(f"   Iteration {iteration}: {active_features} active features")
        
        # Transform back to original scale
        self.weights = self.weights / X_std
        self.bias = y_mean - X_mean @ self.weights
        
        # Store parameters
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        
        # Final metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        active_features = np.sum(np.abs(self.weights) > 1e-8)
        
        l1_penalty_final = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
        l2_penalty_final = self.alpha * (1 - self.l1_ratio) / 2 * np.sum(self.weights**2)
        
        print(f"   MSE: {mse:.6f}, R²: {r2:.6f}")
        print(f"   Active features: {active_features}/{len(self.weights)}")
        print(f"   L1 penalty: {l1_penalty_final:.6f}")
        print(f"   L2 penalty: {l2_penalty_final:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias


# ==========================================
# REGULARIZATION PATH ANALYZER
# ==========================================

class RegularizationPathAnalyzer:
    """
    Analyze and visualize regularization paths for different linear models
    Shows how coefficients change as regularization strength varies
    """
    
    def __init__(self):
        self.paths = {}
        
    def compute_regularization_paths(self, X, y, model_type='ridge', 
                                   alphas=None, l1_ratios=None):
        """
        Compute regularization paths for Ridge, Lasso, or Elastic Net
        """
        if alphas is None:
            alphas = np.logspace(-4, 2, 50)
        
        print(f"\n📈 Computing {model_type.upper()} Regularization Path")
        print(f"   Alpha range: {alphas[0]:.1e} to {alphas[-1]:.1e}")
        print(f"   Number of alpha values: {len(alphas)}")
        
        # Store results
        coefficients = []
        mse_scores = []
        r2_scores = []
        active_features = []
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for i, alpha in enumerate(alphas):
            if model_type == 'ridge':
                model = Ridge(alpha=alpha)
            elif model_type == 'lasso':
                model = Lasso(alpha=alpha, max_iter=2000)
            elif model_type == 'elastic_net':
                if l1_ratios is None:
                    l1_ratio = 0.5
                else:
                    l1_ratio = l1_ratios[i % len(l1_ratios)]
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Store coefficients and metrics
            coefficients.append(model.coef_.copy())
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mse_scores.append(mse)
            r2_scores.append(r2)
            active_features.append(np.sum(np.abs(model.coef_) > 1e-8))
            
            if i % 10 == 0:
                print(f"   α = {alpha:.1e}: {active_features[-1]} active features, R² = {r2:.3f}")
        
        # Store results
        self.paths[model_type] = {
            'alphas': alphas,
            'coefficients': np.array(coefficients),
            'mse_scores': mse_scores,
            'r2_scores': r2_scores,
            'active_features': active_features,
            'best_alpha_idx': np.argmax(r2_scores)
        }
        
        best_alpha = alphas[np.argmax(r2_scores)]
        best_r2 = np.max(r2_scores)
        print(f"   ✅ Best α = {best_alpha:.1e} (R² = {best_r2:.3f})")
        
        return self.paths[model_type]
    
    def plot_regularization_paths(self, model_type='ridge', feature_names=None):
        """
        Visualize regularization paths
        """
        if model_type not in self.paths:
            print(f"No regularization path computed for {model_type}")
            return
        
        path_data = self.paths[model_type]
        alphas = path_data['alphas']
        coefficients = path_data['coefficients']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Coefficient paths
        ax = axes[0, 0]
        for j in range(coefficients.shape[1]):
            if feature_names:
                label = feature_names[j] if j < len(feature_names) else f'Feature {j}'
            else:
                label = f'Feature {j}'
            ax.plot(alphas, coefficients[:, j], label=label, linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'{model_type.upper()} Coefficient Paths')
        ax.grid(True, alpha=0.3)
        if coefficients.shape[1] <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Model performance
        ax = axes[0, 1]
        ax.plot(alphas, path_data['r2_scores'], 'b-', linewidth=2, label='R²')
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance vs Regularization')
        ax.grid(True, alpha=0.3)
        
        # Mark best alpha
        best_idx = path_data['best_alpha_idx']
        ax.axvline(alphas[best_idx], color='red', linestyle='--', alpha=0.7, label='Best α')
        ax.legend()
        
        # 3. Active features
        ax = axes[1, 0]
        ax.plot(alphas, path_data['active_features'], 'g-', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('Number of Active Features')
        ax.set_title('Feature Selection vs Regularization')
        ax.grid(True, alpha=0.3)
        
        # 4. Coefficient magnitude distribution
        ax = axes[1, 1]
        best_coeffs = coefficients[best_idx]
        active_coeffs = best_coeffs[np.abs(best_coeffs) > 1e-8]
        
        if len(active_coeffs) > 0:
            ax.hist(active_coeffs, bins=min(20, len(active_coeffs)), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Coefficient Distribution (Best α = {alphas[best_idx]:.1e})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_regularization_methods(self, X, y, feature_names=None):
        """
        Compare Ridge, Lasso, and Elastic Net on the same dataset
        """
        print(f"\n🔍 Comparing Regularization Methods")
        print("="*50)
        
        # Compute paths for all methods
        ridge_path = self.compute_regularization_paths(X, y, 'ridge')
        lasso_path = self.compute_regularization_paths(X, y, 'lasso')
        elastic_path = self.compute_regularization_paths(X, y, 'elastic_net')
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        methods = ['ridge', 'lasso', 'elastic_net']
        titles = ['Ridge (L2)', 'Lasso (L1)', 'Elastic Net (L1+L2)']
        
        for i, (method, title) in enumerate(zip(methods, titles)):
            path_data = self.paths[method]
            alphas = path_data['alphas']
            coefficients = path_data['coefficients']
            
            # Coefficient paths
            ax = axes[0, i]
            for j in range(min(coefficients.shape[1], 10)):  # Limit to 10 features for clarity
                ax.plot(alphas, coefficients[:, j], linewidth=2)
            ax.set_xscale('log')
            ax.set_xlabel('α')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'{title} Paths')
            ax.grid(True, alpha=0.3)
            
            # Performance comparison
            ax = axes[1, i]
            ax.plot(alphas, path_data['r2_scores'], 'b-', linewidth=2, label='R²')
            ax.plot(alphas, np.array(path_data['active_features'])/coefficients.shape[1], 
                   'r--', linewidth=2, label='Active Features (normalized)')
            ax.set_xscale('log')
            ax.set_xlabel('α')
            ax.set_ylabel('Score')
            ax.set_title(f'{title} Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary comparison
        print(f"\n📊 Best Performance Comparison:")
        for method in methods:
            path_data = self.paths[method]
            best_idx = path_data['best_alpha_idx']
            best_alpha = path_data['alphas'][best_idx]
            best_r2 = path_data['r2_scores'][best_idx]
            best_features = path_data['active_features'][best_idx]
            
            print(f"   {method.upper():<12}: α = {best_alpha:.1e}, R² = {best_r2:.3f}, Features = {best_features}")


# ==========================================
# FEATURE SELECTION FRAMEWORK
# ==========================================

class FeatureSelectionFramework:
    """
    Comprehensive feature selection using various linear model techniques
    """
    
    def __init__(self):
        self.results = {}
        
    def univariate_selection(self, X, y, k=10):
        """
        Select features based on univariate statistical tests
        """
        print(f"\n🎯 Univariate Feature Selection (top {k} features)")
        
        # Compute correlation with target
        correlations = []
        p_values = []
        
        for j in range(X.shape[1]):
            corr = np.corrcoef(X[:, j], y)[0, 1]
            correlations.append(abs(corr))
            
            # Simple t-test for correlation significance
            n = len(X)
            t_stat = corr * np.sqrt((n-2)/(1-corr**2)) if abs(corr) < 0.999 else np.inf
            from scipy.stats import t
            p_val = 2 * (1 - t.cdf(abs(t_stat), n-2))
            p_values.append(p_val)
        
        # Select top k features
        feature_scores = list(zip(range(X.shape[1]), correlations, p_values))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [idx for idx, _, _ in feature_scores[:k]]
        
        print(f"   Selected features: {selected_features}")
        print(f"   Top correlations: {[f'{corr:.3f}' for _, corr, _ in feature_scores[:5]]}")
        
        self.results['univariate'] = {
            'selected_features': selected_features,
            'scores': correlations,
            'p_values': p_values
        }
        
        return selected_features
    
    def lasso_selection(self, X, y, alpha=None):
        """
        Use Lasso regression for automatic feature selection
        """
        print(f"\n🔶 Lasso-based Feature Selection")
        
        if alpha is None:
            # Find optimal alpha using cross-validation
            from sklearn.linear_model import LassoCV
            lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso_cv.fit(X, y)
            alpha = lasso_cv.alpha_
            print(f"   Optimal α found: {alpha:.1e}")
        
        # Fit Lasso with optimal alpha
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X, y)
        
        # Select non-zero coefficients
        selected_features = np.where(np.abs(lasso.coef_) > 1e-8)[0]
        feature_importance = np.abs(lasso.coef_)
        
        print(f"   Selected {len(selected_features)} features out of {X.shape[1]}")
        print(f"   Feature indices: {selected_features.tolist()}")
        
        # Rank by importance
        importance_ranking = np.argsort(feature_importance)[::-1]
        print(f"   Top 5 most important features: {importance_ranking[:5].tolist()}")
        
        self.results['lasso'] = {
            'selected_features': selected_features,
            'coefficients': lasso.coef_,
            'alpha': alpha,
            'importance': feature_importance
        }
        
        return selected_features
    
    def recursive_feature_elimination(self, X, y, n_features=10):
        """
        Recursive Feature Elimination using linear model
        """
        print(f"\n🔄 Recursive Feature Elimination (target: {n_features} features)")
        
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression
        
        # Use RFE with linear regression
        estimator = LinearRegression()
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        selected_features = np.where(rfe.support_)[0]
        feature_ranking = rfe.ranking_
        
        print(f"   Selected features: {selected_features.tolist()}")
        print(f"   Feature ranking (1=best): {feature_ranking[:10].tolist()}...")
        
        self.results['rfe'] = {
            'selected_features': selected_features,
            'ranking': feature_ranking,
            'support': rfe.support_
        }
        
        return selected_features
    
    def compare_selection_methods(self, X, y):
        """
        Compare different feature selection methods
        """
        print(f"\n🔍 Comparing Feature Selection Methods")
        print("="*50)
        
        # Apply all methods
        univariate_features = self.univariate_selection(X, y, k=10)
        lasso_features = self.lasso_selection(X, y)
        rfe_features = self.recursive_feature_elimination(X, y, n_features=10)
        
        # Find overlaps
        all_methods = [
            ('Univariate', set(univariate_features)),
            ('Lasso', set(lasso_features)),
            ('RFE', set(rfe_features))
        ]
        
        print(f"\n🔗 Feature Selection Overlap Analysis:")
        
        # Pairwise overlaps
        for i in range(len(all_methods)):
            for j in range(i+1, len(all_methods)):
                name1, set1 = all_methods[i]
                name2, set2 = all_methods[j]
                overlap = set1.intersection(set2)
                print(f"   {name1} ∩ {name2}: {len(overlap)} features {list(overlap)}")
        
        # Common features across all methods
        common_features = set(univariate_features).intersection(set(lasso_features)).intersection(set(rfe_features))
        print(f"   Common to all methods: {len(common_features)} features {list(common_features)}")
        
        # Visualize selection comparison
        self.plot_selection_comparison()
        
        return {
            'univariate': univariate_features,
            'lasso': lasso_features,
            'rfe': rfe_features,
            'common': list(common_features)
        }
    
    def plot_selection_comparison(self):
        """
        Visualize feature selection results
        """
        if not self.results:
            print("No selection results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Univariate scores
        if 'univariate' in self.results:
            ax = axes[0, 0]
            scores = self.results['univariate']['scores']
            selected = self.results['univariate']['selected_features']
            
            colors = ['red' if i in selected else 'blue' for i in range(len(scores))]
            ax.bar(range(len(scores)), scores, color=colors, alpha=0.7)
            ax.set_title('Univariate Feature Scores')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('|Correlation|')
            ax.grid(True, alpha=0.3)
        
        # 2. Lasso coefficients
        if 'lasso' in self.results:
            ax = axes[0, 1]
            coeffs = self.results['lasso']['coefficients']
            selected = self.results['lasso']['selected_features']
            
            colors = ['red' if i in selected else 'blue' for i in range(len(coeffs))]
            ax.bar(range(len(coeffs)), np.abs(coeffs), color=colors, alpha=0.7)
            ax.set_title('Lasso Coefficient Magnitudes')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('|Coefficient|')
            ax.grid(True, alpha=0.3)
        
        # 3. RFE ranking
        if 'rfe' in self.results:
            ax = axes[1, 0]
            ranking = self.results['rfe']['ranking']
            selected = self.results['rfe']['selected_features']
            
            colors = ['red' if i in selected else 'blue' for i in range(len(ranking))]
            ax.bar(range(len(ranking)), ranking, color=colors, alpha=0.7)
            ax.set_title('RFE Feature Ranking (1=best)')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Rank')
            ax.grid(True, alpha=0.3)
        
        # 4. Selection overlap heatmap
        ax = axes[1, 1]
        methods = ['Univariate', 'Lasso', 'RFE']
        if all(method.lower() in self.results for method in methods):
            overlap_matrix = np.zeros((3, 3))
            
            selections = [
                set(self.results['univariate']['selected_features']),
                set(self.results['lasso']['selected_features']),
                set(self.results['rfe']['selected_features'])
            ]
            
            for i in range(3):
                for j in range(3):
                    if i == j:
                        overlap_matrix[i, j] = len(selections[i])
                    else:
                        overlap_matrix[i, j] = len(selections[i].intersection(selections[j]))
            
            im = ax.imshow(overlap_matrix, cmap='Blues')
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(methods)
            ax.set_yticklabels(methods)
            ax.set_title('Feature Selection Overlap')
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{int(overlap_matrix[i, j])}', 
                           ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()


# ==========================================
# GENERALIZED LINEAR MODELS
# ==========================================

class GeneralizedLinearModels:
    """
    Implementation of various GLMs including logistic regression
    """
    
    def __init__(self):
        self.models = {}
    
    def logistic_regression_from_scratch(self, X, y, learning_rate=0.01, max_iterations=1000):
        """
        Logistic regression implementation using maximum likelihood
        """
        print(f"\n📊 Logistic Regression from Scratch")
        print(f"   Using maximum likelihood estimation")
        print(f"   Learning rate: {learning_rate}")
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        # Training history
        cost_history = []
        accuracy_history = []
        
        def sigmoid(z):
            # Numerically stable sigmoid
            return np.where(z >= 0, 
                          1 / (1 + np.exp(-z)), 
                          np.exp(z) / (1 + np.exp(z)))
        
        for iteration in range(max_iterations):
            # Forward pass
            z = X_with_bias @ theta
            predictions = sigmoid(z)
            
            # Compute cost (negative log-likelihood)
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            cost_history.append(cost)
            
            # Compute accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y)
            accuracy_history.append(accuracy)
            
            # Compute gradients
            gradient = (1 / len(X)) * X_with_bias.T @ (predictions - y)
            
            # Update parameters
            theta -= learning_rate * gradient
            
            # Print progress
            if iteration % 100 == 0:
                print(f"   Iteration {iteration}: Cost = {cost:.4f}, Accuracy = {accuracy:.3f}")
            
            # Check convergence
            if len(cost_history) > 1 and abs(cost_history[-2] - cost_history[-1]) < 1e-8:
                print(f"   ✅ Converged after {iteration + 1} iterations")
                break
        
        self.models['logistic'] = {
            'theta': theta,
            'cost_history': cost_history,
            'accuracy_history': accuracy_history,
            'final_accuracy': accuracy
        }
        
        print(f"   Final accuracy: {accuracy:.3f}")
        
        return theta, cost_history, accuracy_history
    
    def regularized_logistic_regression(self, X, y, reg_type='ridge', alpha=1.0):
        """
        Regularized logistic regression (Ridge/Lasso)
        """
        print(f"\n🔵 Regularized Logistic Regression ({reg_type.upper()})")
        print(f"   Regularization parameter: α = {alpha}")
        
        if reg_type == 'ridge':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(penalty='l2', C=1/alpha, max_iter=2000, random_state=42)
        elif reg_type == 'lasso':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(penalty='l1', C=1/alpha, max_iter=2000, 
                                     random_state=42, solver='liblinear')
        else:
            raise ValueError("reg_type must be 'ridge' or 'lasso'")
        
        # Fit model
        model.fit(X, y)
        
        # Evaluate
        accuracy = model.score(X, y)
        predictions = model.predict(X)
        
        print(f"   Training accuracy: {accuracy:.3f}")
        print(f"   Number of features: {X.shape[1]}")
        
        if hasattr(model, 'coef_'):
            active_features = np.sum(np.abs(model.coef_[0]) > 1e-8)
            print(f"   Active features: {active_features}")
        
        self.models[f'logistic_{reg_type}'] = {
            'model': model,
            'accuracy': accuracy,
            'coefficients': model.coef_[0] if hasattr(model, 'coef_') else None
        }
        
        return model
    
    def compare_glm_regularization(self, X, y):
        """
        Compare different regularization approaches for GLMs
        """
        print(f"\n🔍 Comparing GLM Regularization Methods")
        print("="*50)
        
        # Test different regularization strengths
        alphas = np.logspace(-4, 2, 20)
        
        ridge_scores = []
        lasso_scores = []
        ridge_features = []
        lasso_features = []
        
        for alpha in alphas:
            # Ridge logistic regression
            ridge_model = LogisticRegression(penalty='l2', C=1/alpha, max_iter=2000, random_state=42)
            ridge_scores_cv = cross_val_score(ridge_model, X, y, cv=5)
            ridge_scores.append(np.mean(ridge_scores_cv))
            
            ridge_model.fit(X, y)
            ridge_features.append(np.sum(np.abs(ridge_model.coef_[0]) > 1e-8))
            
            # Lasso logistic regression
            lasso_model = LogisticRegression(penalty='l1', C=1/alpha, max_iter=2000, 
                                           random_state=42, solver='liblinear')
            lasso_scores_cv = cross_val_score(lasso_model, X, y, cv=5)
            lasso_scores.append(np.mean(lasso_scores_cv))
            
            lasso_model.fit(X, y)
            lasso_features.append(np.sum(np.abs(lasso_model.coef_[0]) > 1e-8))
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance comparison
        ax = axes[0]
        ax.plot(alphas, ridge_scores, 'b-', linewidth=2, label='Ridge', marker='o')
        ax.plot(alphas, lasso_scores, 'r-', linewidth=2, label='Lasso', marker='s')
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.set_title('Regularized Logistic Regression Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Feature selection comparison
        ax = axes[1]
        ax.plot(alphas, ridge_features, 'b-', linewidth=2, label='Ridge', marker='o')
        ax.plot(alphas, lasso_features, 'r-', linewidth=2, label='Lasso', marker='s')
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('Number of Active Features')
        ax.set_title('Feature Selection vs Regularization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find best parameters
        best_ridge_idx = np.argmax(ridge_scores)
        best_lasso_idx = np.argmax(lasso_scores)
        
        print(f"\n📊 Best Performance:")
        print(f"   Ridge: α = {alphas[best_ridge_idx]:.1e}, Accuracy = {ridge_scores[best_ridge_idx]:.3f}")
        print(f"   Lasso: α = {alphas[best_lasso_idx]:.1e}, Accuracy = {lasso_scores[best_lasso_idx]:.3f}")
        
        return {
            'alphas': alphas,
            'ridge_scores': ridge_scores,
            'lasso_scores': lasso_scores,
            'ridge_features': ridge_features,
            'lasso_features': lasso_features
        }


# ==========================================
# COMPREHENSIVE DEMONSTRATION
# ==========================================

def comprehensive_linear_models_demo():
    """
    Complete demonstration of linear models deep dive
    """
    print("🎓 Neural Odyssey - Week 14: Linear Models Deep Dive")
    print("=" * 70)
    print("Mastering the Art of Regularized Linear Learning")
    print("=" * 70)
    
    # ================================================================
    # DATA PREPARATION
    # ================================================================
    
    print("\n📊 Preparing Datasets for Comprehensive Analysis")
    
    # 1. Regression dataset with noise and multicollinearity
    X_reg, y_reg = make_regression(n_samples=200, n_features=50, n_informative=10, 
                                  noise=0.1, random_state=42)
    
    # Add some correlated features to demonstrate multicollinearity
    X_corr = np.column_stack([X_reg[:, 0] + np.random.normal(0, 0.1, len(X_reg)),
                             X_reg[:, 1] + np.random.normal(0, 0.1, len(X_reg))])
    X_reg = np.column_stack([X_reg, X_corr])
    
    print(f"   Regression dataset: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
    
    # 2. High-dimensional regression (more features than samples)
    X_high_dim, y_high_dim = make_regression(n_samples=50, n_features=100, 
                                           n_informative=10, noise=0.1, random_state=42)
    print(f"   High-dim dataset: {X_high_dim.shape[0]} samples, {X_high_dim.shape[1]} features")
    
    # 3. Classification dataset
    from sklearn.datasets import make_classification
    X_class, y_class = make_classification(n_samples=500, n_features=20, n_informative=10,
                                         n_redundant=5, random_state=42)
    print(f"   Classification dataset: {X_class.shape[0]} samples, {X_class.shape[1]} features")
    
    # 4. Real-world dataset
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    print(f"   Diabetes dataset: {X_diabetes.shape[0]} samples, {X_diabetes.shape[1]} features")
    
    # ================================================================
    # LINEAR REGRESSION FOUNDATION
    # ================================================================
    
    print("\n" + "="*70)
    print("📐 LINEAR REGRESSION MATHEMATICAL FOUNDATION")
    print("="*70)
    
    # Compare normal equation vs gradient descent
    print("\n1️⃣  Normal Equation vs Gradient Descent Comparison")
    
    lr_normal = LinearRegressionFromScratch(method='normal_equation')
    lr_normal.fit(X_diabetes, y_diabetes)
    
    lr_gd = LinearRegressionFromScratch(method='gradient_descent')
    lr_gd.fit(X_diabetes, y_diabetes, learning_rate=0.01, max_iterations=2000)
    
    print(f"\n   Parameter Comparison:")
    print(f"   Normal Equation - Weights[:3]: {lr_normal.weights[:3]}")
    print(f"   Gradient Descent - Weights[:3]: {lr_gd.weights[:3]}")
    print(f"   Difference: {np.abs(lr_normal.weights[:3] - lr_gd.weights[:3])}")
    
    # Visualize gradient descent convergence
    lr_gd.plot_training_history()
    
    # ================================================================
    # RIDGE REGRESSION ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("🔵 RIDGE REGRESSION DEEP DIVE")
    print("="*70)
    
    print("\n1️⃣  Ridge Regression Implementation and Analysis")
    
    # Compare different alpha values
    alphas_to_test = [0.01, 1.0, 10.0, 100.0]
    
    for alpha in alphas_to_test:
        print(f"\n   Testing α = {alpha}")
        ridge_model = RidgeRegressionFromScratch(alpha=alpha)
        ridge_model.fit(X_diabetes, y_diabetes)
        ridge_model.get_shrinkage_factor(X_diabetes)
    
    # ================================================================
    # LASSO REGRESSION ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("🔶 LASSO REGRESSION DEEP DIVE")
    print("="*70)
    
    print("\n1️⃣  Lasso Feature Selection Demonstration")
    
    # Use high-dimensional dataset to show feature selection
    lasso_model = LassoRegressionFromScratch(alpha=0.1)
    lasso_model.fit(X_high_dim, y_high_dim)
    
    # Visualize regularization path
    lasso_model.plot_regularization_path()
    
    # ================================================================
    # ELASTIC NET ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("🔷 ELASTIC NET DEEP DIVE")
    print("="*70)
    
    print("\n1️⃣  Elastic Net Balancing L1 and L2")
    
    # Test different L1 ratios
    l1_ratios = [0.1, 0.5, 0.9]
    
    for l1_ratio in l1_ratios:
        print(f"\n   Testing L1 ratio = {l1_ratio}")
        elastic_model = ElasticNetFromScratch(alpha=1.0, l1_ratio=l1_ratio)
        elastic_model.fit(X_high_dim, y_high_dim)
    
    # ================================================================
    # REGULARIZATION PATH ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("📈 REGULARIZATION PATH ANALYSIS")
    print("="*70)
    
    path_analyzer = RegularizationPathAnalyzer()
    
    print("\n1️⃣  Computing Regularization Paths")
    
    # Analyze each method
    path_analyzer.compute_regularization_paths(X_diabetes, y_diabetes, 'ridge')
    path_analyzer.plot_regularization_paths('ridge', 
                                          feature_names=[f'Feature_{i}' for i in range(X_diabetes.shape[1])])
    
    path_analyzer.compute_regularization_paths(X_diabetes, y_diabetes, 'lasso')
    path_analyzer.plot_regularization_paths('lasso')
    
    print("\n2️⃣  Comparing All Regularization Methods")
    path_analyzer.compare_regularization_methods(X_diabetes, y_diabetes)
    
    # ================================================================
    # FEATURE SELECTION FRAMEWORK
    # ================================================================
    
    print("\n" + "="*70)
    print("🎯 FEATURE SELECTION FRAMEWORK")
    print("="*70)
    
    feature_selector = FeatureSelectionFramework()
    
    print("\n1️⃣  Comprehensive Feature Selection Comparison")
    
    selection_results = feature_selector.compare_selection_methods(X_class, y_class)
    
    # ================================================================
    # GENERALIZED LINEAR MODELS
    # ================================================================
    
    print("\n" + "="*70)
    print("📊 GENERALIZED LINEAR MODELS")
    print("="*70)
    
    glm_framework = GeneralizedLinearModels()
    
    print("\n1️⃣  Logistic Regression from Scratch")
    
    # Binary classification
    y_binary = (y_class == 1).astype(int)
    theta, cost_history, accuracy_history = glm_framework.logistic_regression_from_scratch(
        X_class, y_binary, learning_rate=0.01, max_iterations=1000
    )
    
    # Plot training progress
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(cost_history, 'b-', linewidth=2)
    axes[0].set_title('Logistic Regression: Cost Function')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Negative Log-Likelihood')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(accuracy_history, 'r-', linewidth=2)
    axes[1].set_title('Logistic Regression: Accuracy')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n2️⃣  Regularized Logistic Regression Comparison")
    
    glm_comparison = glm_framework.compare_glm_regularization(X_class, y_binary)
    
    # ================================================================
    # FINAL INTEGRATION AND INSIGHTS
    # ================================================================
    
    print("\n" + "="*70)
    print("🎓 LINEAR MODELS MASTERY SUMMARY")
    print("="*70)
    
    key_insights = [
        "🔵 Ridge Regression: Shrinks coefficients, handles multicollinearity, preserves all features",
        "🔶 Lasso Regression: Automatic feature selection through L1 penalty, creates sparse solutions", 
        "🔷 Elastic Net: Combines Ridge and Lasso benefits, balances shrinkage and selection",
        "📈 Regularization Paths: Show how coefficients evolve with penalty strength",
        "🎯 Feature Selection: Multiple approaches provide different perspectives on importance",
        "📊 GLMs: Extend linear models to different response distributions and link functions"
    ]
    
    print("\n