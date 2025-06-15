#!/usr/bin/env python3
"""
Neural Odyssey - Phase 2: Core Machine Learning
Week 18: Naive Bayes and Probabilistic Methods
Complete Exercise Implementation

This comprehensive module implements Naive Bayes classifiers from mathematical
first principles, exploring the fundamental connections between probability theory
and machine learning classification.

Learning Path:
1. Bayes' theorem and conditional probability foundations
2. Naive independence assumption and its implications
3. Multiple Naive Bayes variants (Gaussian, Multinomial, Bernoulli)
4. Probabilistic interpretation and decision theory
5. Advanced probabilistic methods and extensions
6. Real-world applications and practical considerations

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional, Union, Dict, List, Any
import warnings
from collections import defaultdict, Counter
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from sklearn.datasets import make_classification, fetch_20newsgroups, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import re

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

class BayesianFoundations:
    """
    Foundational Bayesian methods and probability theory implementations
    """
    
    @staticmethod
    def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
        """
        Apply Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
        
        Parameters:
        -----------
        prior : P(H) - prior probability of hypothesis
        likelihood : P(E|H) - probability of evidence given hypothesis
        evidence : P(E) - marginal probability of evidence
        
        Returns:
        --------
        posterior : P(H|E) - posterior probability
        """
        if evidence == 0:
            return 0.0
        return (likelihood * prior) / evidence
    
    @staticmethod
    def log_bayes_theorem(log_prior: float, log_likelihood: float, log_evidence: float) -> float:
        """
        Bayes' theorem in log space for numerical stability
        """
        return log_prior + log_likelihood - log_evidence
    
    @staticmethod
    def marginal_likelihood(likelihoods: np.ndarray, priors: np.ndarray) -> float:
        """
        Calculate marginal likelihood (evidence): P(E) = Σ P(E|H_i) * P(H_i)
        """
        return np.sum(likelihoods * priors)
    
    @staticmethod
    def posterior_distribution(likelihoods: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """
        Calculate full posterior distribution over all hypotheses
        """
        evidence = BayesianFoundations.marginal_likelihood(likelihoods, priors)
        if evidence == 0:
            return np.zeros_like(priors)
        return (likelihoods * priors) / evidence

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features
    
    Assumes features follow Gaussian distributions within each class
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes
        
        Parameters:
        -----------
        var_smoothing : float, smoothing parameter for variance calculation
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = None
        self.feature_means_ = None
        self.feature_vars_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit Gaussian Naive Bayes classifier
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store basic information
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate class priors
        self.class_priors_ = {}
        for class_label in self.classes_:
            class_count = np.sum(y == class_label)
            self.class_priors_[class_label] = class_count / n_samples
        
        # Calculate feature statistics for each class
        self.feature_means_ = {}
        self.feature_vars_ = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Calculate mean and variance for each feature
            self.feature_means_[class_label] = np.mean(X_class, axis=0)
            self.feature_vars_[class_label] = np.var(X_class, axis=0) + self.var_smoothing
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: float, var: float) -> np.ndarray:
        """Calculate Gaussian probability density function"""
        coefficient = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = -0.5 * ((x - mean) ** 2) / var
        return coefficient * np.exp(exponent)
    
    def _log_gaussian_pdf(self, x: np.ndarray, mean: float, var: float) -> np.ndarray:
        """Calculate log Gaussian PDF for numerical stability"""
        log_coefficient = -0.5 * np.log(2 * np.pi * var)
        log_exponent = -0.5 * ((x - mean) ** 2) / var
        return log_coefficient + log_exponent
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log class probabilities
        
        Returns log probabilities to avoid numerical underflow
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probabilities = np.zeros((n_samples, n_classes))
        
        for class_idx, class_label in enumerate(self.classes_):
            # Log prior
            log_prior = np.log(self.class_priors_[class_label])
            
            # Log likelihood for each feature (assuming independence)
            log_likelihood = 0
            for feature_idx in range(self.n_features_):
                mean = self.feature_means_[class_label][feature_idx]
                var = self.feature_vars_[class_label][feature_idx]
                
                feature_log_likelihood = self._log_gaussian_pdf(
                    X[:, feature_idx], mean, var
                )
                log_likelihood += feature_log_likelihood
            
            # Log posterior (unnormalized)
            log_probabilities[:, class_idx] = log_prior + log_likelihood
        
        # Normalize using log-sum-exp trick
        log_probabilities = log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)
        
        return log_probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_proba = self.predict_log_proba(X)
        class_indices = np.argmax(log_proba, axis=1)
        return self.classes_[class_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score"""
        return accuracy_score(y, self.predict(X))

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes for discrete features (e.g., text classification)
    
    Assumes features follow multinomial distributions
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes
        
        Parameters:
        -----------
        alpha : float, additive smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_log_probs_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit Multinomial Naive Bayes classifier
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store basic information
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate class priors with smoothing
        self.class_priors_ = {}
        for class_label in self.classes_:
            class_count = np.sum(y == class_label)
            # Add-one smoothing for priors
            self.class_priors_[class_label] = (class_count + self.alpha) / (n_samples + len(self.classes_) * self.alpha)
        
        # Calculate feature log probabilities for each class
        self.feature_log_probs_ = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Sum of feature counts for this class
            feature_counts = np.sum(X_class, axis=0)
            
            # Total count for normalization
            total_count = np.sum(feature_counts)
            
            # Calculate smoothed probabilities
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + self.n_features_ * self.alpha
            
            # Store log probabilities
            self.feature_log_probs_[class_label] = np.log(smoothed_counts / smoothed_total)
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log class probabilities"""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probabilities = np.zeros((n_samples, n_classes))
        
        for class_idx, class_label in enumerate(self.classes_):
            # Log prior
            log_prior = np.log(self.class_priors_[class_label])
            
            # Log likelihood: sum over features weighted by their counts
            log_likelihood = X @ self.feature_log_probs_[class_label]
            
            # Log posterior (unnormalized)
            log_probabilities[:, class_idx] = log_prior + log_likelihood
        
        # Normalize
        log_probabilities = log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)
        
        return log_probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_proba = self.predict_log_proba(X)
        class_indices = np.argmax(log_proba, axis=1)
        return self.classes_[class_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score"""
        return accuracy_score(y, self.predict(X))

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes for binary features
    
    Assumes features are binary (0 or 1) and follow Bernoulli distributions
    """
    
    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        """
        Initialize Bernoulli Naive Bayes
        
        Parameters:
        -----------
        alpha : float, additive smoothing parameter
        binarize : float or None, threshold for binarizing features
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_priors_ = None
        self.feature_log_probs_ = None
        self.n_features_ = None
        
    def _binarize_features(self, X: np.ndarray) -> np.ndarray:
        """Binarize features if threshold is set"""
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        """Fit Bernoulli Naive Bayes classifier"""
        X = np.array(X)
        y = np.array(y)
        
        # Binarize features if necessary
        X = self._binarize_features(X)
        
        # Store basic information
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate class priors
        self.class_priors_ = {}
        for class_label in self.classes_:
            class_count = np.sum(y == class_label)
            self.class_priors_[class_label] = (class_count + self.alpha) / (n_samples + len(self.classes_) * self.alpha)
        
        # Calculate feature probabilities for each class
        self.feature_log_probs_ = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            n_class_samples = np.sum(class_mask)
            
            # Calculate probability of each feature being 1 in this class
            feature_counts = np.sum(X_class, axis=0)
            
            # Smoothed probabilities
            prob_feature_1 = (feature_counts + self.alpha) / (n_class_samples + 2 * self.alpha)
            prob_feature_0 = 1 - prob_feature_1
            
            # Store log probabilities for both 0 and 1
            self.feature_log_probs_[class_label] = {
                'log_prob_1': np.log(prob_feature_1),
                'log_prob_0': np.log(prob_feature_0)
            }
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log class probabilities"""
        X = np.array(X)
        X = self._binarize_features(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probabilities = np.zeros((n_samples, n_classes))
        
        for class_idx, class_label in enumerate(self.classes_):
            # Log prior
            log_prior = np.log(self.class_priors_[class_label])
            
            # Log likelihood for each sample
            log_likelihood = np.zeros(n_samples)
            
            for feature_idx in range(self.n_features_):
                feature_values = X[:, feature_idx]
                
                # Add log probability based on feature value (0 or 1)
                prob_1 = self.feature_log_probs_[class_label]['log_prob_1'][feature_idx]
                prob_0 = self.feature_log_probs_[class_label]['log_prob_0'][feature_idx]
                
                log_likelihood += np.where(feature_values == 1, prob_1, prob_0)
            
            # Log posterior (unnormalized)
            log_probabilities[:, class_idx] = log_prior + log_likelihood
        
        # Normalize
        log_probabilities = log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)
        
        return log_probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_proba = self.predict_log_proba(X)
        class_indices = np.argmax(log_proba, axis=1)
        return self.classes_[class_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score"""
        return accuracy_score(y, self.predict(X))

class NaiveBayesEnsemble:
    """
    Ensemble of different Naive Bayes variants with automatic model selection
    """
    
    def __init__(self, models: Optional[List] = None):
        """
        Initialize Naive Bayes ensemble
        
        Parameters:
        -----------
        models : list of Naive Bayes models, if None uses default set
        """
        if models is None:
            self.models = [
                ('Gaussian', GaussianNaiveBayes()),
                ('Multinomial', MultinomialNaiveBayes()),
                ('Bernoulli', BernoulliNaiveBayes())
            ]
        else:
            self.models = models
        
        self.best_model_ = None
        self.best_model_name_ = None
        self.model_scores_ = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> 'NaiveBayesEnsemble':
        """
        Fit ensemble and select best model based on validation performance
        """
        X = np.array(X)
        y = np.array(y)
        
        # Split data for model selection
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        best_score = -1
        
        for model_name, model in self.models:
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                score = model.score(X_val, y_val)
                self.model_scores_[model_name] = score
                
                # Update best model
                if score > best_score:
                    best_score = score
                    self.best_model_ = model
                    self.best_model_name_ = model_name
                    
            except Exception as e:
                print(f"Warning: {model_name} failed to fit: {e}")
                self.model_scores_[model_name] = 0.0
        
        # Refit best model on full data
        if self.best_model_ is not None:
            self.best_model_.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best model"""
        if self.best_model_ is None:
            raise ValueError("Ensemble not fitted yet!")
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using best model"""
        if self.best_model_ is None:
            raise ValueError("Ensemble not fitted yet!")
        return self.best_model_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score"""
        if self.best_model_ is None:
            raise ValueError("Ensemble not fitted yet!")
        return self.best_model_.score(X, y)

class TextNaiveBayes:
    """
    Specialized Naive Bayes for text classification with preprocessing
    """
    
    def __init__(self, 
                 vectorizer_type: str = 'count',
                 max_features: Optional[int] = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 alpha: float = 1.0):
        """
        Initialize Text Naive Bayes
        
        Parameters:
        -----------
        vectorizer_type : str, 'count' or 'tfidf'
        max_features : int, maximum number of features
        ngram_range : tuple, n-gram range for feature extraction
        alpha : float, smoothing parameter
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        
        # Initialize vectorizer
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
        
        # Initialize Naive Bayes classifier
        self.classifier = MultinomialNaiveBayes(alpha=alpha)
        
    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess text data"""
        processed = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            processed.append(text)
        
        return processed
    
    def fit(self, texts: List[str], y: np.ndarray) -> 'TextNaiveBayes':
        """
        Fit text classifier
        
        Parameters:
        -----------
        texts : list of strings
        y : array-like of labels
        """
        # Preprocess texts
        processed_texts = self._preprocess_text(texts)
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Convert to dense array for our implementation
        X_dense = X.toarray()
        
        # Fit Naive Bayes classifier
        self.classifier.fit(X_dense, y)
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts"""
        processed_texts = self._preprocess_text(texts)
        X = self.vectorizer.transform(processed_texts)
        X_dense = X.toarray()
        return self.classifier.predict(X_dense)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for texts"""
        processed_texts = self._preprocess_text(texts)
        X = self.vectorizer.transform(processed_texts)
        X_dense = X.toarray()
        return self.classifier.predict_proba(X_dense)
    
    def score(self, texts: List[str], y: np.ndarray) -> float:
        """Return accuracy score"""
        return accuracy_score(y, self.predict(texts))
    
    def get_feature_importance(self, class_label: int, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get most important features for a class
        
        Returns:
        --------
        List of (feature_name, log_probability) tuples
        """
        feature_names = self.vectorizer.get_feature_names_out()
        class_probs = self.classifier.feature_log_probs_[class_label]
        
        # Get top k features
        top_indices = np.argsort(class_probs)[-top_k:][::-1]
        
        return [(feature_names[i], class_probs[i]) for i in top_indices]

class NaiveBayesAnalyzer:
    """
    Comprehensive analysis toolkit for Naive Bayes classifiers
    """
    
    @staticmethod
    def plot_feature_distributions(X: np.ndarray, y: np.ndarray, 
                                  feature_names: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (15, 10)):
        """Plot feature distributions by class"""
        n_features = min(X.shape[1], 12)  # Limit to 12 features for visualization
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
        
        for i in range(n_features):
            ax = axes[i]
            
            for class_idx, class_label in enumerate(classes):
                class_data = X[y == class_label, i]
                
                ax.hist(class_data, alpha=0.6, label=f'Class {class_label}', 
                       color=colors[class_idx], bins=20, density=True)
            
            feature_name = feature_names[i] if feature_names else f'Feature {i}'
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(clf, X: np.ndarray, y: np.ndarray,
                             feature_indices: Tuple[int, int] = (0, 1),
                             title: str = "Naive Bayes Decision Boundary",
                             figsize: Tuple[int, int] = (10, 8)):
        """Plot decision boundary for 2D feature space"""
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features for decision boundary plot")
        
        plt.figure(figsize=figsize)
        
        # Extract two features
        X_2d = X[:, list(feature_indices)]
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create full feature space for prediction (fill others with mean)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        if X.shape[1] > 2:
            # For higher dimensional data, set other features to their mean
            full_mesh = np.zeros((mesh_points.shape[0], X.shape[1]))
            full_mesh[:, feature_indices[0]] = mesh_points[:, 0]
            full_mesh[:, feature_indices[1]] = mesh_points[:, 1]
            
            for i in range(X.shape[1]):
                if i not in feature_indices:
                    full_mesh[:, i] = np.mean(X[:, i])
            
            mesh_points = full_mesh
        
        # Predict on mesh
        Z = clf.predict_proba(mesh_points)[:, 1] if len(clf.classes_) == 2 else clf.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.viridis,
                            edgecolors='black', alpha=0.7)
        
        plt.colorbar(scatter)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def analyze_independence_assumption(X: np.ndarray, y: np.ndarray,
                                      feature_names: Optional[List[str]] = None,
                                      figsize: Tuple[int, int] = (12, 8)):
        """Analyze the naive independence assumption"""
        print("NAIVE INDEPENDENCE ASSUMPTION ANALYSIS")
        print("="*50)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        
        # Plot correlation heatmap
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Feature Correlation Matrix')
        
        # Calculate and display independence violations
        plt.subplot(1, 2, 2)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = abs(correlation_matrix[i, j])
                if corr > 0.5:  # Threshold for high correlation
                    feature_i = feature_names[i] if feature_names else f'F{i}'
                    feature_j = feature_names[j] if feature_names else f'F{j}'
                    high_corr_pairs.append((feature_i, feature_j, corr))
        
        if high_corr_pairs:
            pairs, correlations = zip(*[(f"{pair[0]}-{pair[1]}", pair[2]) for pair in high_corr_pairs])
            plt.barh(range(len(pairs)), correlations, color='red', alpha=0.7)
            plt.yticks(range(len(pairs)), pairs)
            plt.xlabel('Absolute Correlation')
            plt.title('High Correlation Pairs (|r| > 0.5)')
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        else:
            plt.text(0.5, 0.5, 'No high correlations found\n(independence assumption holds)',
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        plt.tight_layout()
        plt.show()
        
        # Statistical independence tests
        print(f"\nFeature Independence Analysis:")
        print(f"Number of feature pairs with |correlation| > 0.5: {len(high_corr_pairs)}")
        print(f"Maximum absolute correlation: {np.max(np.abs(correlation_matrix[mask == 0])):.3f}")
        
        if high_corr_pairs:
            print("\nTop correlated pairs:")
            for pair in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
                print(f"  {pair[0]} - {pair[1]}: r = {pair[2]:.3f}")
        
        return correlation_matrix, high_corr_pairs
    
    @staticmethod
    def compare_nb_variants(X: np.ndarray, y: np.ndarray,
                           test_size: float = 0.2,
                           figsize: Tuple[int, int] = (12, 8)):
        """Compare different Naive Bayes variants"""
        print("\nNAIVE BAYES VARIANTS COMPARISON")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Initialize models
        models = {
            'Gaussian NB': GaussianNaiveBayes(),
            'Multinomial NB': MultinomialNaiveBayes(alpha=1.0),
            'Bernoulli NB': BernoulliNaiveBayes(alpha=1.0)
        }
        
        results = {}
        
        # Evaluate each model
        for name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = []
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in kf.split(X, y):
                    model_cv = type(model)(**model.__dict__)
                    model_cv.fit(X[train_idx], y[train_idx])
                    cv_scores.append(model_cv.score(X[val_idx], y[val_idx]))
                
                results[name] = {
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'model': model
                }
                
                print(f"{name:15} | Train: {train_score:.3f} | Test: {test_score:.3f} | CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                
            except Exception as e:
                print(f"{name:15} | Failed: {str(e)}")
                results[name] = None
        
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Performance comparison
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            model_names = list(valid_results.keys())
            train_scores = [valid_results[name]['train_score'] for name in model_names]
            test_scores = [valid_results[name]['test_score'] for name in model_names]
            cv_means = [valid_results[name]['cv_mean'] for name in model_names]
            cv_stds = [valid_results[name]['cv_std'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.25
            
            ax1.bar(x - width, train_scores, width, label='Train', alpha=0.8)
            ax1.bar(x, test_scores, width, label='Test', alpha=0.8)
            ax1.bar(x + width, cv_means, width, label='CV Mean', alpha=0.8)
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Cross-validation variance
            ax2.errorbar(model_names, cv_means, yerr=cv_stds, fmt='o-', capsize=5, capthick=2)
            ax2.set_ylabel('CV Accuracy')
            ax2.set_title('Cross-Validation Performance')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results

class NaiveBayesExperiments:
    """Comprehensive Naive Bayes experiments and case studies"""
    
    @staticmethod
    def experiment_1_gaussian_nb_basics():
        """Experiment 1: Gaussian Naive Bayes fundamentals"""
        print("="*70)
        print("EXPERIMENT 1: GAUSSIAN NAIVE BAYES FUNDAMENTALS")
        print("="*70)
        
        # Generate synthetic dataset with different distributions
        np.random.seed(42)
        
        # Class 0: Lower means
        X0 = np.random.multivariate_normal([2, 3], [[1, 0.3], [0.3, 1]], 150)
        y0 = np.zeros(150)
        
        # Class 1: Higher means
        X1 = np.random.multivariate_normal([6, 7], [[1, -0.4], [-0.4, 1]], 150)
        y1 = np.ones(150)
        
        # Combine data
        X_gauss = np.vstack([X0, X1])
        y_gauss = np.hstack([y0, y1])
        
        print("\n1.1 BASIC GAUSSIAN NAIVE BAYES")
        print("-" * 50)
        
        # Train Gaussian Naive Bayes
        gnb = GaussianNaiveBayes()
        gnb.fit(X_gauss, y_gauss)
        
        print(f"Training Accuracy: {gnb.score(X_gauss, y_gauss):.3f}")
        print(f"Number of classes: {len(gnb.classes_)}")
        print(f"Class priors: {gnb.class_priors_}")
        
        # Analyze learned parameters
        print("\nLearned Parameters:")
        for class_label in gnb.classes_:
            print(f"Class {int(class_label)}:")
            print(f"  Mean: {gnb.feature_means_[class_label]}")
            print(f"  Variance: {gnb.feature_vars_[class_label]}")
        
        # Visualize decision boundary
        NaiveBayesAnalyzer.plot_decision_boundary(gnb, X_gauss, y_gauss,
                                                "Gaussian Naive Bayes Decision Boundary")
        
        # Analyze feature distributions
        NaiveBayesAnalyzer.plot_feature_distributions(X_gauss, y_gauss, 
                                                    ['Feature 1', 'Feature 2'])
        
        print("\n1.2 PROBABILITY PREDICTIONS")
        print("-" * 50)
        
        # Test probability predictions
        test_points = np.array([[1, 2], [4, 5], [7, 8]])
        probabilities = gnb.predict_proba(test_points)
        predictions = gnb.predict(test_points)
        
        print("Test Point Predictions:")
        for i, (point, pred, proba) in enumerate(zip(test_points, predictions, probabilities)):
            print(f"Point {point}: Class {int(pred)}, P(Class 0) = {proba[0]:.3f}, P(Class 1) = {proba[1]:.3f}")
        
        return {
            'data': (X_gauss, y_gauss),
            'model': gnb,
            'test_results': (test_points, predictions, probabilities)
        }
    
    @staticmethod
    def experiment_2_text_classification():
        """Experiment 2: Text classification with Multinomial Naive Bayes"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: TEXT CLASSIFICATION WITH MULTINOMIAL NAIVE BAYES")
        print("="*70)
        
        # Create sample text dataset
        documents = [
            "machine learning algorithms are powerful tools",
            "neural networks can learn complex patterns",
            "deep learning revolutionizes artificial intelligence",
            "data science involves statistics and programming",
            "python is great for machine learning projects",
            "cars need regular maintenance and care",
            "driving safely requires attention and skill",
            "automotive technology advances rapidly",
            "electric vehicles are environmentally friendly",
            "traffic congestion is a major urban problem"
        ]
        
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0: tech, 1: automotive
        
        print("\n2.1 TEXT PREPROCESSING AND VECTORIZATION")
        print("-" * 50)
        
        # Initialize text classifier
        text_nb = TextNaiveBayes(vectorizer_type='count', max_features=50, ngram_range=(1, 1))
        text_nb.fit(documents, labels)
        
        print(f"Vocabulary size: {len(text_nb.vectorizer.get_feature_names_out())}")
        print(f"Training accuracy: {text_nb.score(documents, labels):.3f}")
        
        # Show feature importance
        print("\n2.2 MOST IMPORTANT FEATURES BY CLASS")
        print("-" * 50)
        
        for class_label in [0, 1]:
            class_name = "Technology" if class_label == 0 else "Automotive"
            important_features = text_nb.get_feature_importance(class_label, top_k=10)
            
            print(f"\nTop features for {class_name}:")
            for feature, log_prob in important_features:
                print(f"  {feature}: {np.exp(log_prob):.4f}")
        
        # Test on new documents
        test_docs = [
            "artificial intelligence and machine learning",
            "car maintenance and automotive repair",
            "programming with python and data analysis"
        ]
        
        predictions = text_nb.predict(test_docs)
        probabilities = text_nb.predict_proba(test_docs)
        
        print("\n2.3 PREDICTIONS ON NEW DOCUMENTS")
        print("-" * 50)
        
        class_names = ["Technology", "Automotive"]
        for i, (doc, pred, proba) in enumerate(zip(test_docs, predictions, probabilities)):
            print(f"\nDocument: '{doc}'")
            print(f"Predicted class: {class_names[pred]}")
            print(f"Probabilities: Tech={proba[0]:.3f}, Auto={proba[1]:.3f}")
        
        return {
            'classifier': text_nb,
            'test_docs': test_docs,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    @staticmethod
    def experiment_3_binary_features():
        """Experiment 3: Bernoulli Naive Bayes for binary features"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: BERNOULLI NAIVE BAYES FOR BINARY FEATURES")
        print("="*70)
        
        # Generate binary feature dataset (e.g., presence/absence of symptoms)
        np.random.seed(42)
        
        # Simulate medical diagnosis dataset
        n_samples = 1000
        n_features = 10
        
        # Class 0: Healthy (lower probability of symptoms)
        X0 = np.random.binomial(1, 0.2, (n_samples//2, n_features))
        y0 = np.zeros(n_samples//2)
        
        # Class 1: Sick (higher probability of symptoms)
        X1 = np.random.binomial(1, 0.7, (n_samples//2, n_features))
        y1 = np.ones(n_samples//2)
        
        X_binary = np.vstack([X0, X1])
        y_binary = np.hstack([y0, y1])
        
        feature_names = [f'Symptom_{i+1}' for i in range(n_features)]
        
        print("\n3.1 BERNOULLI NAIVE BAYES ON BINARY DATA")
        print("-" * 50)
        
        # Train Bernoulli Naive Bayes
        bnb = BernoulliNaiveBayes(alpha=1.0)
        bnb.fit(X_binary, y_binary)
        
        print(f"Training Accuracy: {bnb.score(X_binary, y_binary):.3f}")
        
        # Analyze learned probabilities
        print("\nLearned Feature Probabilities:")
        for class_label in bnb.classes_:
            class_name = "Healthy" if class_label == 0 else "Sick"
            print(f"\n{class_name} (Class {int(class_label)}):")
            
            prob_1 = np.exp(bnb.feature_log_probs_[class_label]['log_prob_1'])
            
            for i, (feature, p) in enumerate(zip(feature_names, prob_1)):
                print(f"  P({feature}=1|{class_name}) = {p:.3f}")
        
        # Cross-validation evaluation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_binary, y_binary):
            bnb_cv = BernoulliNaiveBayes(alpha=1.0)
            bnb_cv.fit(X_binary[train_idx], y_binary[train_idx])
            cv_scores.append(bnb_cv.score(X_binary[val_idx], y_binary[val_idx]))
        
        print(f"\nCross-validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Test predictions
        test_cases = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Many symptoms
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No symptoms
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]   # Few symptoms
        ])
        
        predictions = bnb.predict(test_cases)
        probabilities = bnb.predict_proba(test_cases)
        
        print("\n3.2 DIAGNOSTIC PREDICTIONS")
        print("-" * 50)
        
        for i, (case, pred, proba) in enumerate(zip(test_cases, predictions, probabilities)):
            symptoms_present = np.sum(case)
            diagnosis = "Healthy" if pred == 0 else "Sick"
            
            print(f"\nCase {i+1}: {symptoms_present} symptoms present")
            print(f"Symptoms: {case}")
            print(f"Diagnosis: {diagnosis}")
            print(f"Confidence: P(Healthy)={proba[0]:.3f}, P(Sick)={proba[1]:.3f}")
        
        return {
            'data': (X_binary, y_binary),
            'model': bnb,
            'feature_names': feature_names,
            'cv_scores': cv_scores
        }
    
    @staticmethod
    def experiment_4_model_comparison():
        """Experiment 4: Comprehensive model comparison"""
        print("\n" + "="*70)
        print("EXPERIMENT 4: NAIVE BAYES MODEL COMPARISON")
        print("="*70)
        
        # Load wine dataset for comparison
        wine = load_wine()
        X_wine, y_wine = wine.data, wine.target
        
        print("\n4.1 DATASET ANALYSIS")
        print("-" * 50)
        
        print(f"Dataset: Wine classification")
        print(f"Samples: {X_wine.shape[0]}")
        print(f"Features: {X_wine.shape[1]}")
        print(f"Classes: {len(np.unique(y_wine))}")
        print(f"Class distribution: {np.bincount(y_wine)}")
        
        # Analyze independence assumption
        correlation_matrix, high_corr_pairs = NaiveBayesAnalyzer.analyze_independence_assumption(
            X_wine, y_wine, wine.feature_names
        )
        
        print("\n4.2 MODEL COMPARISON")
        print("-" * 50)
        
        # Compare different Naive Bayes variants
        comparison_results = NaiveBayesAnalyzer.compare_nb_variants(X_wine, y_wine)
        
        # Ensemble comparison
        print("\n4.3 ENSEMBLE MODEL")
        print("-" * 50)
        
        ensemble = NaiveBayesEnsemble()
        ensemble.fit(X_wine, y_wine)
        
        print(f"Best model: {ensemble.best_model_name_}")
        print(f"Model scores: {ensemble.model_scores_}")
        
        # Detailed evaluation on test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
        )
        
        ensemble.fit(X_train, y_train)
        test_accuracy = ensemble.score(X_test, y_test)
        
        print(f"Ensemble test accuracy: {test_accuracy:.3f}")
        
        # Classification report
        y_pred = ensemble.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=wine.target_names))
        
        return {
            'data': (X_wine, y_wine),
            'comparison_results': comparison_results,
            'ensemble': ensemble,
            'test_accuracy': test_accuracy
        }
    
    @staticmethod
    def experiment_5_real_world_newsgroups():
        """Experiment 5: Real-world text classification on 20 newsgroups"""
        print("\n" + "="*70)
        print("EXPERIMENT 5: REAL-WORLD TEXT CLASSIFICATION (20 NEWSGROUPS)")
        print("="*70)
        
        # Load subset of 20 newsgroups dataset
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        
        print("\n5.1 DATA LOADING AND PREPROCESSING")
        print("-" * 50)
        
        try:
            newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                                 shuffle=True, random_state=42,
                                                 remove=('headers', 'footers', 'quotes'))
            newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                                shuffle=True, random_state=42,
                                                remove=('headers', 'footers', 'quotes'))
            
            print(f"Training samples: {len(newsgroups_train.data)}")
            print(f"Test samples: {len(newsgroups_test.data)}")
            print(f"Categories: {newsgroups_train.target_names}")
            
            # Train text classifier
            text_classifier = TextNaiveBayes(
                vectorizer_type='tfidf',
                max_features=5000,
                ngram_range=(1, 2),
                alpha=0.1
            )
            
            print("\n5.2 TRAINING TEXT CLASSIFIER")
            print("-" * 50)
            
            text_classifier.fit(newsgroups_train.data, newsgroups_train.target)
            
            # Evaluate
            train_accuracy = text_classifier.score(newsgroups_train.data, newsgroups_train.target)
            test_accuracy = text_classifier.score(newsgroups_test.data, newsgroups_test.target)
            
            print(f"Training accuracy: {train_accuracy:.3f}")
            print(f"Test accuracy: {test_accuracy:.3f}")
            
            # Analyze most informative features
            print("\n5.3 MOST INFORMATIVE FEATURES")
            print("-" * 50)
            
            for class_idx, category in enumerate(newsgroups_train.target_names):
                print(f"\nTop features for '{category}':")
                important_features = text_classifier.get_feature_importance(class_idx, top_k=10)
                for feature, score in important_features:
                    print(f"  {feature}: {np.exp(score):.4f}")
            
            # Test on sample documents
            sample_texts = [
                "I believe in God and follow Christian teachings",
                "Computer graphics and image processing algorithms",
                "Medical research shows promising results for treatment",
                "There is no evidence for the existence of God"
            ]
            
            predictions = text_classifier.predict(sample_texts)
            probabilities = text_classifier.predict_proba(sample_texts)
            
            print("\n5.4 SAMPLE PREDICTIONS")
            print("-" * 50)
            
            for i, (text, pred, proba) in enumerate(zip(sample_texts, predictions, probabilities)):
                predicted_category = newsgroups_train.target_names[pred]
                max_prob = np.max(proba)
                
                print(f"\nText: '{text}'")
                print(f"Predicted: {predicted_category} (confidence: {max_prob:.3f})")
                
                # Show all probabilities
                for j, category in enumerate(newsgroups_train.target_names):
                    print(f"  P({category}) = {proba[j]:.3f}")
            
            # Detailed evaluation
            y_pred = text_classifier.predict(newsgroups_test.data)
            
            print("\n5.5 DETAILED EVALUATION")
            print("-" * 50)
            print(classification_report(newsgroups_test.target, y_pred, 
                                      target_names=newsgroups_train.target_names))
            
            # Confusion matrix
            cm = confusion_matrix(newsgroups_test.target, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=newsgroups_train.target_names,
                       yticklabels=newsgroups_train.target_names)
            plt.title('Confusion Matrix - 20 Newsgroups Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
            
            return {
                'classifier': text_classifier,
                'train_data': newsgroups_train,
                'test_data': newsgroups_test,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"Could not load 20 newsgroups dataset: {e}")
            print("Skipping this experiment...")
            return None
    
    @staticmethod
    def experiment_6_probabilistic_insights():
        """Experiment 6: Deep dive into probabilistic insights"""
        print("\n" + "="*70)
        print("EXPERIMENT 6: PROBABILISTIC INSIGHTS AND THEORETICAL ANALYSIS")
        print("="*70)
        
        # Generate controlled dataset to demonstrate concepts
        np.random.seed(42)
        
        # Create dataset with known properties
        n_samples = 1000
        
        # Feature 1: Strongly predictive
        X1_class0 = np.random.normal(0, 1, n_samples//2)
        X1_class1 = np.random.normal(3, 1, n_samples//2)
        
        # Feature 2: Moderately predictive
        X2_class0 = np.random.normal(0, 1.5, n_samples//2)
        X2_class1 = np.random.normal(2, 1.5, n_samples//2)
        
        # Feature 3: Weakly predictive
        X3_class0 = np.random.normal(0, 2, n_samples//2)
        X3_class1 = np.random.normal(1, 2, n_samples//2)
        
        # Feature 4: Non-predictive (noise)
        X4 = np.random.normal(0, 1, n_samples)
        
        X_insights = np.column_stack([
            np.hstack([X1_class0, X1_class1]),
            np.hstack([X2_class0, X2_class1]),
            np.hstack([X3_class0, X3_class1]),
            X4
        ])
        
        y_insights = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        feature_names = ['Strong_Predictor', 'Moderate_Predictor', 'Weak_Predictor', 'Noise']
        
        print("\n6.1 BAYESIAN REASONING ANALYSIS")
        print("-" * 50)
        
        # Train Gaussian Naive Bayes
        gnb_insights = GaussianNaiveBayes()
        gnb_insights.fit(X_insights, y_insights)
        
        print(f"Overall accuracy: {gnb_insights.score(X_insights, y_insights):.3f}")
        
        # Analyze each feature's discriminative power
        print("\nFeature Discriminative Analysis:")
        for i, feature_name in enumerate(feature_names):
            # Calculate separability measure (difference in means / pooled std)
            mean_0 = gnb_insights.feature_means_[0][i]
            mean_1 = gnb_insights.feature_means_[1][i]
            var_0 = gnb_insights.feature_vars_[0][i]
            var_1 = gnb_insights.feature_vars_[1][i]
            
            pooled_std = np.sqrt((var_0 + var_1) / 2)
            separability = abs(mean_1 - mean_0) / pooled_std
            
            print(f"  {feature_name}:")
            print(f"    Class 0: μ={mean_0:.2f}, σ²={var_0:.2f}")
            print(f"    Class 1: μ={mean_1:.2f}, σ²={var_1:.2f}")
            print(f"    Separability: {separability:.2f}")
        
        print("\n6.2 POSTERIOR PROBABILITY ANALYSIS")
        print("-" * 50)
        
        # Analyze how probabilities change with evidence
        test_point = np.array([[0, 0, 0, 0]])  # Neutral point
        
        print("Posterior probabilities as evidence accumulates:")
        print("Base case (no evidence): P(Class 0) = 0.5, P(Class 1) = 0.5")
        
        # Add evidence feature by feature
        for i in range(len(feature_names)):
            partial_point = np.zeros((1, len(feature_names)))
            partial_point[0, :i+1] = test_point[0, :i+1]
            
            # Create temporary model with only first i+1 features
            gnb_partial = GaussianNaiveBayes()
            gnb_partial.fit(X_insights[:, :i+1], y_insights)
            
            proba = gnb_partial.predict_proba(partial_point[:, :i+1])[0]
            
            print(f"After observing {feature_names[i]}: P(Class 0) = {proba[0]:.3f}, P(Class 1) = {proba[1]:.3f}")
        
        print("\n6.3 UNCERTAINTY QUANTIFICATION")
        print("-" * 50)
        
        # Analyze prediction confidence across the feature space
        probabilities = gnb_insights.predict_proba(X_insights)
        max_probabilities = np.max(probabilities, axis=1)
        
        # Classify predictions by confidence
        high_confidence = max_probabilities > 0.9
        medium_confidence = (max_probabilities > 0.7) & (max_probabilities <= 0.9)
        low_confidence = max_probabilities <= 0.7
        
        print(f"High confidence predictions (>0.9): {np.sum(high_confidence)/len(X_insights)*100:.1f}%")
        print(f"Medium confidence predictions (0.7-0.9): {np.sum(medium_confidence)/len(X_insights)*100:.1f}%")
        print(f"Low confidence predictions (<0.7): {np.sum(low_confidence)/len(X_insights)*100:.1f}%")
        
        # Accuracy by confidence level
        predictions = gnb_insights.predict(X_insights)
        
        if np.sum(high_confidence) > 0:
            high_conf_accuracy = accuracy_score(y_insights[high_confidence], predictions[high_confidence])
            print(f"Accuracy on high confidence predictions: {high_conf_accuracy:.3f}")
        
        if np.sum(low_confidence) > 0:
            low_conf_accuracy = accuracy_score(y_insights[low_confidence], predictions[low_confidence])
            print(f"Accuracy on low confidence predictions: {low_conf_accuracy:.3f}")
        
        # Visualize confidence distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(max_probabilities, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Maximum Posterior Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(x=0.7, color='orange', linestyle='--', label='Medium threshold')
        plt.axvline(x=0.9, color='red', linestyle='--', label='High threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance visualization
        plt.subplot(2, 2, 2)
        separabilities = []
        for i in range(len(feature_names)):
            mean_0 = gnb_insights.feature_means_[0][i]
            mean_1 = gnb_insights.feature_means_[1][i]
            var_0 = gnb_insights.feature_vars_[0][i]
            var_1 = gnb_insights.feature_vars_[1][i]
            pooled_std = np.sqrt((var_0 + var_1) / 2)
            separability = abs(mean_1 - mean_0) / pooled_std
            separabilities.append(separability)
        
        plt.bar(feature_names, separabilities, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Features')
        plt.ylabel('Separability (|μ₁-μ₀|/σ)')
        plt.title('Feature Discriminative Power')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Decision boundary for top 2 features
        plt.subplot(2, 2, 3)
        top_features = np.argsort(separabilities)[-2:]
        X_top2 = X_insights[:, top_features]
        
        # Create mesh for decision boundary
        h = 0.1
        x_min, x_max = X_top2[:, 0].min() - 1, X_top2[:, 0].max() + 1
        y_min, y_max = X_top2[:, 1].min() - 1, X_top2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Train model on top 2 features
        gnb_2d = GaussianNaiveBayes()
        gnb_2d.fit(X_top2, y_insights)
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = gnb_2d.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        scatter = plt.scatter(X_top2[:, 0], X_top2[:, 1], c=y_insights, 
                            cmap='viridis', alpha=0.7, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel(f'{feature_names[top_features[0]]}')
        plt.ylabel(f'{feature_names[top_features[1]]}')
        plt.title('Decision Boundary (Top 2 Features)')
        
        # Prior vs Posterior comparison
        plt.subplot(2, 2, 4)
        
        # Calculate likelihood ratios for a range of values
        x_range = np.linspace(-4, 6, 100)
        feature_idx = 0  # Use strongest feature
        
        mean_0 = gnb_insights.feature_means_[0][feature_idx]
        mean_1 = gnb_insights.feature_means_[1][feature_idx]
        var_0 = gnb_insights.feature_vars_[0][feature_idx]
        var_1 = gnb_insights.feature_vars_[1][feature_idx]
        
        # Calculate likelihoods
        likelihood_0 = gnb_insights._gaussian_pdf(x_range, mean_0, var_0)
        likelihood_1 = gnb_insights._gaussian_pdf(x_range, mean_1, var_1)
        
        plt.plot(x_range, likelihood_0, label='P(x|Class 0)', alpha=0.7, linewidth=2)
        plt.plot(x_range, likelihood_1, label='P(x|Class 1)', alpha=0.7, linewidth=2)
        
        # Mark intersection point
        intersection_idx = np.argmin(np.abs(likelihood_0 - likelihood_1))
        plt.axvline(x=x_range[intersection_idx], color='red', linestyle='--', 
                   label=f'Decision boundary: x={x_range[intersection_idx]:.2f}')
        
        plt.xlabel(f'{feature_names[feature_idx]} Value')
        plt.ylabel('Likelihood')
        plt.title('Likelihood Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'data': (X_insights, y_insights),
            'model': gnb_insights,
            'feature_names': feature_names,
            'separabilities': separabilities,
            'confidence_stats': {
                'high_confidence': np.sum(high_confidence),
                'medium_confidence': np.sum(medium_confidence),
                'low_confidence': np.sum(low_confidence)
            }
        }

class NaiveBayesTutorial:
    """Interactive Naive Bayes tutorial and conceptual explanations"""
    
    @staticmethod
    def bayes_theorem_tutorial():
        """Interactive tutorial on Bayes' theorem"""
        print("="*80)
        print("BAYES' THEOREM TUTORIAL")
        print("="*80)
        
        print("""
        Bayes' Theorem: The Foundation of Probabilistic Reasoning
        ========================================================
        
        Formula: P(H|E) = P(E|H) × P(H) / P(E)
        
        Where:
        - P(H|E) = Posterior probability (what we want to find)
        - P(E|H) = Likelihood (how likely is the evidence given the hypothesis)
        - P(H) = Prior probability (our initial belief)
        - P(E) = Evidence or marginal likelihood (normalizing constant)
        
        Key Insights:
        1. We start with a prior belief P(H)
        2. We observe evidence E
        3. We update our belief to get the posterior P(H|E)
        4. The likelihood P(E|H) determines how much the evidence supports our hypothesis
        """)
        
        # Medical diagnosis example
        print("\nClassic Example: Medical Diagnosis")
        print("-" * 40)
        
        # Disease prevalence
        prior_disease = 0.01  # 1% of population has the disease
        
        # Test characteristics
        sensitivity = 0.95    # P(positive test | disease) = 95%
        specificity = 0.95    # P(negative test | no disease) = 95%
        false_positive_rate = 1 - specificity  # 5%
        
        # Calculate marginal likelihood
        evidence = (sensitivity * prior_disease + 
                   false_positive_rate * (1 - prior_disease))
        
        # Apply Bayes' theorem
        posterior = BayesianFoundations.bayes_theorem(
            prior_disease, sensitivity, evidence
        )
        
        print(f"Disease prevalence (prior): {prior_disease*100:.1f}%")
        print(f"Test sensitivity: {sensitivity*100:.1f}%")
        print(f"Test specificity: {specificity*100:.1f}%")
        print(f"P(positive test): {evidence*100:.1f}%")
        print(f"P(disease | positive test): {posterior*100:.1f}%")
        
        print(f"\nKey Insight: Even with a 95% accurate test, a positive result")
        print(f"only indicates {posterior*100:.1f}% probability of disease!")
        print(f"This is because the disease is rare (low prior).")
        
        # Demonstrate how prior affects posterior
        print("\nEffect of Different Priors:")
        print("-" * 30)
        
        priors = [0.001, 0.01, 0.1, 0.5]  # Different disease prevalences
        
        for prior in priors:
            evidence_i = sensitivity * prior + false_positive_rate * (1 - prior)
            posterior_i = BayesianFoundations.bayes_theorem(prior, sensitivity, evidence_i)
            print(f"Prior: {prior*100:5.1f}% → Posterior: {posterior_i*100:5.1f}%")
    
    @staticmethod
    def naive_independence_tutorial():
        """Tutorial on the naive independence assumption"""
        print("\n" + "="*80)
        print("NAIVE INDEPENDENCE ASSUMPTION TUTORIAL")
        print("="*80)
        
        print("""
        The "Naive" Assumption in Naive Bayes
        ====================================
        
        Naive Bayes assumes that features are conditionally independent given the class:
        
        P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
        
        This is "naive" because in reality, features are often correlated.
        
        Why does it still work well?
        1. Even with violated assumptions, the relative ranking of classes often remains correct
        2. The decision boundary can still be effective even if probability estimates are off
        3. Many real-world datasets have features that are approximately independent
        4. The simplicity of the model provides good generalization (bias-variance tradeoff)
        """)
        
        # Demonstrate with synthetic example
        print("\nDemonstration: Correlated vs Independent Features")
        print("-" * 50)
        
        np.random.seed(42)
        n_samples = 1000
        
        # Independent features
        X1_indep = np.random.normal(0, 1, (n_samples, 2))
        X1_indep[n_samples//2:, :] += [2, 2]  # Shift class 1
        y_indep = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Correlated features (violates naive assumption)
        mean0, cov0 = [0, 0], [[1, 0.8], [0.8, 1]]
        mean1, cov1 = [2, 2], [[1, 0.8], [0.8, 1]]
        
        X0_corr = np.random.multivariate_normal(mean0, cov0, n_samples//2)
        X1_corr = np.random.multivariate_normal(mean1, cov1, n_samples//2)
        X_corr = np.vstack([X0_corr, X1_corr])
        y_corr = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Train models
        nb_indep = GaussianNaiveBayes()
        nb_corr = GaussianNaiveBayes()
        
        nb_indep.fit(X1_indep, y_indep)
        nb_corr.fit(X_corr, y_corr)
        
        acc_indep = nb_indep.score(X1_indep, y_indep)
        acc_corr = nb_corr.score(X_corr, y_corr)
        
        print(f"Independent features - Accuracy: {acc_indep:.3f}")
        print(f"Correlated features - Accuracy: {acc_corr:.3f}")
        
        # Calculate actual correlation
        corr_indep = np.corrcoef(X1_indep.T)[0, 1]
        corr_corr = np.corrcoef(X_corr.T)[0, 1]
        
        print(f"Feature correlation (independent): {corr_indep:.3f}")
        print(f"Feature correlation (correlated): {corr_corr:.3f}")
        
        print(f"\nInsight: Even with strong correlation ({corr_corr:.3f}),")
        print(f"Naive Bayes still achieves {acc_corr:.3f} accuracy!")
    
    @staticmethod
    def smoothing_tutorial():
        """Tutorial on smoothing techniques"""
        print("\n" + "="*80)
        print("SMOOTHING TECHNIQUES TUTORIAL")
        print("="*80)
        
        print("""
        Why Smoothing is Necessary
        =========================
        
        Problem: Zero probabilities
        - If a feature value never appears with a particular class in training data
        - P(feature=value|class) = 0
        - This makes the entire posterior probability 0 (multiplication by zero)
        - New data might contain previously unseen feature values
        
        Solution: Additive Smoothing (Laplace Smoothing)
        - Add a small constant α to all counts
        - For discrete features: P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × |vocabulary|)
        - For continuous features: Add small value to variance estimates
        
        Effect of Smoothing Parameter α:
        - α = 0: No smoothing (risk of zero probabilities)
        - α = 1: Laplace smoothing (uniform pseudocounts)
        - α > 1: Stronger smoothing (more conservative estimates)
        - α < 1: Lighter smoothing
        """)
        
        # Demonstrate with text example
        print("\nDemonstration: Text Classification with Smoothing")
        print("-" * 50)
        
        # Small text dataset to show zero probability problem
        small_docs = [
            "machine learning is great",
            "python programming is fun",
            "data science uses statistics",
            "cars need regular maintenance",
            "driving requires skill"
        ]
        
        small_labels = [0, 0, 0, 1, 1]  # 0: tech, 1: automotive
        
        # Test document with unseen word
        test_doc = ["artificial intelligence is amazing"]
        
        print("Training documents:")
        for i, (doc, label) in enumerate(zip(small_docs, small_labels)):
            category = "Tech" if label == 0 else "Auto"
            print(f"  {category}: '{doc}'")
        
        print(f"\nTest document: '{test_doc[0]}'")
        print("Note: 'artificial', 'intelligence', 'amazing' are unseen words")
        
        # Compare different smoothing values
        alphas = [0.01, 0.1, 1.0, 10.0]
        
        print(f"\nEffect of Smoothing Parameter:")
        print(f"{'Alpha':>8} {'Prediction':>12} {'Max Prob':>10}")
        print("-" * 32)
        
        for alpha in alphas:
            try:
                text_nb = TextNaiveBayes(alpha=alpha, max_features=20)
                text_nb.fit(small_docs, np.array(small_labels))
                
                pred = text_nb.predict(test_doc)[0]
                proba = text_nb.predict_proba(test_doc)[0]
                max_prob = np.max(proba)
                
                category = "Tech" if pred == 0 else "Auto"
                print(f"{alpha:>8.2f} {category:>12} {max_prob:>10.3f}")
                
            except Exception as e:
                print(f"{alpha:>8.2f} {'Failed':>12} {'N/A':>10}")
        
        print(f"\nInsights:")
        print(f"- Very small α may cause numerical issues with unseen words")
        print(f"- Large α provides more conservative (less confident) predictions")
        print(f"- α = 1.0 (Laplace smoothing) is often a good default")
    
    @staticmethod
    def practical_guidelines():
        """Practical guidelines for using Naive Bayes"""
        print("\n" + "="*80)
        print("PRACTICAL GUIDELINES FOR NAIVE BAYES")
        print("="*80)
        
        guidelines = [
            {
                'title': 'When to Use Naive Bayes',
                'points': [
                    "Text classification (especially with many features)",
                    "Real-time predictions (very fast inference)",
                    "Small datasets (works well with limited data)",
                    "Baseline model (quick to implement and interpret)",
                    "Multi-class problems (naturally handles multiple classes)",
                    "When features are actually close to independent"
                ]
            },
            {
                'title': 'When NOT to Use Naive Bayes',
                'points': [
                    "Highly correlated features with strong dependence",
                    "When you need well-calibrated probability estimates",
                    "Complex non-linear relationships between features",
                    "When you have very few features (<5)",
                    "Time series data (violates independence assumption)",
                    "When interpretability is not important and accuracy is key"
                ]
            },
            {
                'title': 'Feature Engineering for Naive Bayes',
                'points': [
                    "Remove highly correlated features",
                    "Use feature selection to identify most informative features",
                    "For text: consider n-grams, TF-IDF, or word embeddings",
                    "For continuous features: consider discretization",
                    "Normalize features if using Gaussian NB",
                    "Handle missing values explicitly"
                ]
            },
            {
                'title': 'Hyperparameter Tuning',
                'points': [
                    "Smoothing parameter α: start with 1.0, tune on validation set",
                    "For text: tune vocabulary size and n-gram range",
                    "For continuous data: consider different variance estimators",
                    "Use cross-validation for parameter selection",
                    "Consider ensemble of different NB variants",
                    "Monitor for overfitting with very small α values"
                ]
            },
            {
                'title': 'Performance Optimization',
                'points': [
                    "Use log probabilities to avoid numerical underflow",
                    "Implement incremental learning for streaming data",
                    "Vectorize operations for better performance",
                    "Consider sparse matrices for text data",
                    "Pre-compute log priors and feature statistics",
                    "Use appropriate data structures for large vocabularies"
                ]
            }
        ]
        
        for guideline in guidelines:
            print(f"\n{guideline['title']}")
            print("-" * len(guideline['title']))
            for point in guideline['points']:
                print(f"  • {point}")
        
        # Performance comparison table
        print(f"\nNaive Bayes Variants Comparison:")
        print("-" * 60)
        print(f"{'Variant':15} {'Data Type':15} {'Use Case':25} {'Speed':10}")
        print("-" * 60)
        print(f"{'Gaussian':15} {'Continuous':15} {'General classification':25} {'Fast':10}")
        print(f"{'Multinomial':15} {'Discrete/Count':15} {'Text classification':25} {'Fast':10}")
        print(f"{'Bernoulli':15} {'Binary':15} {'Binary features':25} {'Fast':10}")
        print(f"{'Complement':15} {'Discrete/Count':15} {'Imbalanced text':25} {'Fast':10}")

def main():
    """Main function to run all Naive Bayes experiments and tutorials"""
    print("🧠 NEURAL ODYSSEY - WEEK 18: NAIVE BAYES AND PROBABILISTIC METHODS")
    print("="*80)
    print("Complete implementation of Naive Bayes from probabilistic first principles")
    print("="*80)
    
    # Foundational tutorials
    NaiveBayesTutorial.bayes_theorem_tutorial()
    NaiveBayesTutorial.naive_independence_tutorial()
    NaiveBayesTutorial.smoothing_tutorial()
    
    # Run all experiments
    experiments_results = {}
    
    # Experiment 1: Gaussian Naive Bayes basics
    experiments_results['gaussian_basics'] = NaiveBayesExperiments.experiment_1_gaussian_nb_basics()
    
    # Experiment 2: Text classification
    experiments_results['text_classification'] = NaiveBayesExperiments.experiment_2_text_classification()
    
    # Experiment 3: Binary features
    experiments_results['binary_features'] = NaiveBayesExperiments.experiment_3_binary_features()
    
    # Experiment 4: Model comparison
    experiments_results['model_comparison'] = NaiveBayesExperiments.experiment_4_model_comparison()
    
    # Experiment 5: Real-world newsgroups
    experiments_results['newsgroups'] = NaiveBayesExperiments.experiment_5_real_world_newsgroups()
    
    # Experiment 6: Probabilistic insights
    experiments_results['probabilistic_insights'] = NaiveBayesExperiments.experiment_6_probabilistic_insights()
    
    # Practical guidelines
    NaiveBayesTutorial.practical_guidelines()
    
    # Final summary
    print("\n" + "="*80)
    print("WEEK 18 SUMMARY: NAIVE BAYES AND PROBABILISTIC METHODS MASTERY")
    print("="*80)
    
    summary_text = """
    🎯 ACHIEVEMENTS UNLOCKED:
    
    ✅ Bayesian Foundations
       • Mastered Bayes' theorem and conditional probability
       • Understood prior, likelihood, and posterior relationships
       • Applied probabilistic reasoning to classification problems
    
    ✅ Naive Bayes Variants
       • Implemented Gaussian NB for continuous features
       • Built Multinomial NB for discrete/count data
       • Created Bernoulli NB for binary features
       • Developed ensemble methods for automatic model selection
    
    ✅ Text Classification Mastery
       • Specialized text preprocessing and vectorization
       • N-gram feature extraction and TF-IDF weighting
       • Real-world application on 20 newsgroups dataset
       • Feature importance analysis and interpretation
    
    ✅ Probabilistic Insights
       • Analyzed the naive independence assumption
       • Understood smoothing techniques and their effects
       • Quantified prediction uncertainty and confidence
       • Connected theory to practical decision making
    
    ✅ Advanced Applications
       • Multi-class classification strategies
       • Handling imbalanced datasets
       • Real-time prediction systems
       • Integration with modern ML pipelines
    
    🔗 CONNECTIONS TO BROADER ML:
       • Probabilistic foundation for many ML algorithms
       • Base classifier in ensemble methods
       • Feature selection using mutual information
       • Calibration techniques for probability estimates
       • Online learning and streaming data processing
    
    📈 NEXT STEPS:
       • Explore advanced probabilistic models (Bayesian networks)
       • Study connections to information theory and entropy
       • Apply to recommendation systems and collaborative filtering
       • Investigate deep probabilistic models and variational inference
    """
    
    print(summary_text)
    
    return {
        'experiments': experiments_results,
        'status': 'completed'
    }

if __name__ == "__main__":
    # Set up plotting for better visualization
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # Run complete Naive Bayes learning module
    results = main()
    
    print("\n🎉 Week 18: Naive Bayes and Probabilistic Methods - COMPLETED!")
    print("Ready to advance to Week 19: Decision Trees and Information Theory")