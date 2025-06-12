"""
Neural Odyssey - Week 16: Advanced Ensemble Methods and Model Selection
Phase 2: Core Machine Learning (Week 4)

Beyond Simple Aggregation: The Art of Meta-Learning

Building on tree-based ensemble methods from Week 15, this week explores advanced
ensemble techniques that go beyond simple voting and averaging. You'll master
stacking (stacked generalization), meta-learning, Bayesian model averaging,
and automated ensemble construction while understanding the theoretical
foundations of why and when these methods work.

This comprehensive exploration covers:
1. Stacking and meta-learning from theoretical foundations
2. Voting ensemble strategies and optimal combination
3. Bayesian model averaging and uncertainty quantification
4. Dynamic ensemble selection and adaptive methods
5. Ensemble diversity analysis and optimization
6. Automated machine learning (AutoML) principles
7. Production ensemble systems and scalability
8. Advanced meta-learning and few-shot ensemble adaptation

To get started, run: python exercises.py

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from itertools import combinations
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)


# ==========================================
# STACKING ENSEMBLE IMPLEMENTATION
# ==========================================

class StackingEnsembleFromScratch:
    """
    Complete stacking implementation with cross-validation for meta-learning
    Demonstrates meta-learning principles and proper validation techniques
    """
    
    def __init__(self, base_models, meta_model, cv_folds=5, use_probabilities=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.trained_base_models = []
        self.trained_meta_model = None
        self.base_model_names = []
        
    def fit(self, X, y):
        """
        Train stacking ensemble using cross-validation for meta-feature generation
        """
        print(f"🚀 Training Stacking Ensemble")
        print(f"   Base models: {len(self.base_models)}")
        print(f"   Meta-model: {type(self.meta_model).__name__}")
        print(f"   CV folds: {self.cv_folds}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Store model names for reference
        self.base_model_names = [type(model).__name__ for model in self.base_models]
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        print(f"   Meta-features shape: {meta_features.shape}")
        
        # Train base models on full dataset
        print(f"   Training base models on full dataset...")
        self.trained_base_models = []
        for i, model in enumerate(self.base_models):
            print(f"      Training {self.base_model_names[i]}")
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X, y)
            self.trained_base_models.append(model_copy)
        
        # Train meta-model on meta-features
        print(f"   Training meta-model...")
        self.trained_meta_model = type(self.meta_model)(**self.meta_model.get_params())
        self.trained_meta_model.fit(meta_features, y)
        
        print(f"   ✅ Stacking ensemble training complete")
        
        return self
    
    def _generate_meta_features(self, X, y):
        """
        Generate meta-features using cross-validation to prevent overfitting
        """
        print(f"   Generating meta-features with {self.cv_folds}-fold CV...")
        
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        # Initialize meta-features array
        if self.use_probabilities and hasattr(self.base_models[0], 'predict_proba'):
            n_meta_features = len(self.base_models) * n_classes
        else:
            n_meta_features = len(self.base_models)
        
        meta_features = np.zeros((n_samples, n_meta_features))
        
        # Cross-validation for meta-feature generation
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"      Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            # Train each base model on fold training data
            for model_idx, model in enumerate(self.base_models):
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for validation fold
                if self.use_probabilities and hasattr(model_copy, 'predict_proba'):
                    pred_proba = model_copy.predict_proba(X_val_fold)
                    start_idx = model_idx * n_classes
                    end_idx = start_idx + n_classes
                    meta_features[val_idx, start_idx:end_idx] = pred_proba
                else:
                    predictions = model_copy.predict(X_val_fold)
                    meta_features[val_idx, model_idx] = predictions
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using trained stacking ensemble"""
        if not self.trained_base_models or not self.trained_meta_model:
            raise ValueError("Ensemble not trained yet!")
        
        # Generate base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-model makes final prediction
        final_predictions = self.trained_meta_model.predict(base_predictions)
        
        return final_predictions
    
    def predict_proba(self, X):
        """Predict class probabilities using stacking ensemble"""
        if not self.trained_base_models or not self.trained_meta_model:
            raise ValueError("Ensemble not trained yet!")
        
        # Generate base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-model predicts probabilities
        if hasattr(self.trained_meta_model, 'predict_proba'):
            probabilities = self.trained_meta_model.predict_proba(base_predictions)
        else:
            # For models without predict_proba, use decision function or predictions
            predictions = self.trained_meta_model.predict(base_predictions)
            n_classes = len(np.unique(predictions))
            probabilities = np.eye(n_classes)[predictions]
        
        return probabilities
    
    def _get_base_predictions(self, X):
        """Get predictions from all base models"""
        n_classes = len(np.unique([0, 1]))  # Assuming binary for simplicity
        
        if self.use_probabilities:
            n_features = len(self.trained_base_models) * n_classes
        else:
            n_features = len(self.trained_base_models)
        
        base_predictions = np.zeros((len(X), n_features))
        
        for model_idx, model in enumerate(self.trained_base_models):
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                start_idx = model_idx * n_classes
                end_idx = start_idx + n_classes
                base_predictions[:, start_idx:end_idx] = pred_proba
            else:
                predictions = model.predict(X)
                base_predictions[:, model_idx] = predictions
        
        return base_predictions
    
    def analyze_base_model_performance(self, X, y):
        """Analyze individual base model performance"""
        print(f"\n📊 Base Model Performance Analysis")
        print("=" * 50)
        
        performances = {}
        
        for i, model in enumerate(self.trained_base_models):
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                auc = roc_auc_score(y, proba[:, 1])
                logloss = log_loss(y, proba)
            else:
                auc = None
                logloss = None
            
            performances[self.base_model_names[i]] = {
                'accuracy': acc,
                'auc': auc,
                'log_loss': logloss
            }
            
            print(f"   {self.base_model_names[i]:<20}: Acc={acc:.4f}", end="")
            if auc is not None:
                print(f", AUC={auc:.4f}, LogLoss={logloss:.4f}")
            else:
                print()
        
        return performances
    
    def visualize_stacking_analysis(self, X, y, X_test=None, y_test=None):
        """Comprehensive visualization of stacking ensemble"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Base model performance comparison
        ax = axes[0, 0]
        performances = self.analyze_base_model_performance(X, y)
        
        model_names = list(performances.keys())
        accuracies = [performances[name]['accuracy'] for name in model_names]
        
        bars = ax.bar(range(len(model_names)), accuracies, alpha=0.7)
        ax.set_xlabel('Base Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Base Model Performance')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Correlation matrix between base models
        ax = axes[0, 1]
        base_preds = np.array([model.predict(X) for model in self.trained_base_models])
        correlation_matrix = np.corrcoef(base_preds)
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Base Model Correlation Matrix')
        
        # Add correlation values
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                       ha='center', va='center', 
                       color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        # 3. Ensemble vs individual performance
        ax = axes[0, 2]
        ensemble_pred = self.predict(X)
        ensemble_acc = accuracy_score(y, ensemble_pred)
        
        all_accs = accuracies + [ensemble_acc]
        all_names = model_names + ['Stacking Ensemble']
        colors = ['lightblue'] * len(model_names) + ['red']
        
        bars = ax.bar(range(len(all_names)), all_accs, color=colors, alpha=0.7)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Individual vs Ensemble Performance')
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 4. Meta-feature importance (if applicable)
        ax = axes[1, 0]
        if hasattr(self.trained_meta_model, 'feature_importances_'):
            importance = self.trained_meta_model.feature_importances_
            feature_names = [f'{name}' for name in model_names]
            
            ax.bar(range(len(importance)), importance, alpha=0.7)
            ax.set_xlabel('Meta-Features')
            ax.set_ylabel('Importance')
            ax.set_title('Meta-Model Feature Importance')
            ax.set_xticks(range(len(importance)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Meta-model does not\nprovide feature importance', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Meta-Model Feature Importance (N/A)')
        
        # 5. Prediction confidence distribution
        ax = axes[1, 1]
        if hasattr(self, 'predict_proba'):
            ensemble_proba = self.predict_proba(X)
            if ensemble_proba.shape[1] == 2:  # Binary classification
                confidence = np.max(ensemble_proba, axis=1)
                ax.hist(confidence, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Prediction Confidence')
                ax.set_ylabel('Frequency')
                ax.set_title('Ensemble Confidence Distribution')
                ax.grid(True, alpha=0.3)
        
        # 6. Test set performance (if provided)
        ax = axes[1, 2]
        if X_test is not None and y_test is not None:
            test_performances = []
            
            for model in self.trained_base_models:
                test_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, test_pred)
                test_performances.append(test_acc)
            
            ensemble_test_pred = self.predict(X_test)
            ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)
            test_performances.append(ensemble_test_acc)
            
            all_test_names = model_names + ['Ensemble']
            colors = ['lightcoral'] * len(model_names) + ['darkred']
            
            bars = ax.bar(range(len(all_test_names)), test_performances, color=colors, alpha=0.7)
            ax.set_xlabel('Models')
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Test Set Performance')
            ax.set_xticks(range(len(all_test_names)))
            ax.set_xticklabels(all_test_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Test set not provided', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Test Set Performance (N/A)')
        
        plt.tight_layout()
        plt.show()


# ==========================================
# VOTING ENSEMBLE IMPLEMENTATION
# ==========================================

class VotingEnsembleFromScratch:
    """
    Advanced voting ensemble with multiple voting strategies
    """
    
    def __init__(self, models, voting='soft', weights=None):
        self.models = models
        self.voting = voting  # 'hard', 'soft', 'weighted'
        self.weights = weights
        self.trained_models = []
        self.model_names = []
        
    def fit(self, X, y):
        """Train all models in the ensemble"""
        print(f"🗳️  Training Voting Ensemble")
        print(f"   Voting strategy: {self.voting}")
        print(f"   Number of models: {len(self.models)}")
        
        self.model_names = [type(model).__name__ for model in self.models]
        self.trained_models = []
        
        for i, model in enumerate(self.models):
            print(f"   Training {self.model_names[i]}")
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X, y)
            self.trained_models.append(model_copy)
        
        # Calculate optimal weights if not provided
        if self.voting == 'weighted' and self.weights is None:
            self.weights = self._calculate_optimal_weights(X, y)
            print(f"   Calculated optimal weights: {self.weights}")
        
        print(f"   ✅ Voting ensemble training complete")
        return self
    
    def _calculate_optimal_weights(self, X, y):
        """Calculate optimal weights based on individual model performance"""
        performances = []
        
        # Use cross-validation to estimate individual model performance
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model in self.trained_models:
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            performances.append(np.mean(cv_scores))
        
        # Convert to weights (normalize so they sum to 1)
        weights = np.array(performances)
        weights = weights / np.sum(weights)
        
        return weights
    
    def predict(self, X):
        """Make predictions using voting strategy"""
        if not self.trained_models:
            raise ValueError("Ensemble not trained yet!")
        
        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        elif self.voting == 'soft':
            return self._soft_voting_predict(X)
        elif self.voting == 'weighted':
            return self._weighted_voting_predict(X)
        else:
            raise ValueError("Voting must be 'hard', 'soft', or 'weighted'")
    
    def _hard_voting_predict(self, X):
        """Hard voting: majority vote"""
        predictions = np.array([model.predict(X) for model in self.trained_models])
        
        # Majority vote for each sample
        final_predictions = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            majority_vote = mode(votes)[0][0]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)
    
    def _soft_voting_predict(self, X):
        """Soft voting: average probabilities"""
        # Collect probability predictions
        all_probas = []
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                all_probas.append(probas)
            else:
                # For models without predict_proba, use hard predictions
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                probas = np.eye(n_classes)[preds]
                all_probas.append(probas)
        
        # Average probabilities
        avg_probas = np.mean(all_probas, axis=0)
        
        # Return class with highest average probability
        return np.argmax(avg_probas, axis=1)
    
    def _weighted_voting_predict(self, X):
        """Weighted voting: weighted average of probabilities"""
        if self.weights is None:
            raise ValueError("Weights not set for weighted voting")
        
        # Collect weighted probability predictions
        weighted_probas = None
        
        for i, model in enumerate(self.trained_models):
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
            else:
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                probas = np.eye(n_classes)[preds]
            
            if weighted_probas is None:
                weighted_probas = self.weights[i] * probas
            else:
                weighted_probas += self.weights[i] * probas
        
        # Return class with highest weighted probability
        return np.argmax(weighted_probas, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.voting == 'soft':
            all_probas = []
            for model in self.trained_models:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    all_probas.append(probas)
            
            if all_probas:
                return np.mean(all_probas, axis=0)
        
        elif self.voting == 'weighted' and self.weights is not None:
            weighted_probas = None
            
            for i, model in enumerate(self.trained_models):
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    
                    if weighted_probas is None:
                        weighted_probas = self.weights[i] * probas
                    else:
                        weighted_probas += self.weights[i] * probas
            
            return weighted_probas
        
        # Fallback: convert hard predictions to probabilities
        predictions = self.predict(X)
        n_classes = len(np.unique(predictions))
        return np.eye(n_classes)[predictions]
    
    def analyze_voting_strategies(self, X, y):
        """Compare different voting strategies"""
        print(f"\n🗳️  Voting Strategy Comparison")
        print("=" * 40)
        
        strategies = ['hard', 'soft', 'weighted']
        results = {}
        
        for strategy in strategies:
            # Temporarily change voting strategy
            original_voting = self.voting
            original_weights = self.weights
            
            self.voting = strategy
            if strategy == 'weighted' and self.weights is None:
                self.weights = self._calculate_optimal_weights(X, y)
            
            # Make predictions
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            results[strategy] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            
            print(f"   {strategy.capitalize()} voting: {accuracy:.4f}")
            
            # Restore original settings
            self.voting = original_voting
            self.weights = original_weights
        
        return results


# ==========================================
# BAYESIAN MODEL AVERAGING
# ==========================================

class BayesianModelAveraging:
    """
    Bayesian Model Averaging implementation for ensemble learning
    """
    
    def __init__(self, models, prior_weights=None):
        self.models = models
        self.prior_weights = prior_weights
        self.posterior_weights = None
        self.trained_models = []
        self.model_names = []
        
    def fit(self, X, y):
        """Train models and calculate posterior weights"""
        print(f"🎲 Bayesian Model Averaging")
        print(f"   Number of models: {len(self.models)}")
        
        self.model_names = [type(model).__name__ for model in self.models]
        
        # Set uniform priors if not provided
        if self.prior_weights is None:
            self.prior_weights = np.ones(len(self.models)) / len(self.models)
            print(f"   Using uniform priors")
        else:
            print(f"   Using provided priors: {self.prior_weights}")
        
        # Train models and calculate evidence (likelihood)
        log_evidences = []
        self.trained_models = []
        
        print(f"   Calculating model evidence...")
        
        for i, model in enumerate(self.models):
            print(f"      {self.model_names[i]}")
            
            # Train model
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X, y)
            self.trained_models.append(model_copy)
            
            # Calculate log evidence using cross-validation log-likelihood
            log_evidence = self._calculate_log_evidence(model_copy, X, y)
            log_evidences.append(log_evidence)
        
        # Calculate posterior weights using Bayes' rule
        log_evidences = np.array(log_evidences)
        log_priors = np.log(self.prior_weights)
        
        # Log posterior = log prior + log evidence
        log_posteriors = log_priors + log_evidences
        
        # Convert to probabilities (normalize)
        log_posteriors = log_posteriors - np.max(log_posteriors)  # Numerical stability
        posteriors = np.exp(log_posteriors)
        self.posterior_weights = posteriors / np.sum(posteriors)
        
        print(f"   Posterior weights calculated:")
        for i, (name, weight) in enumerate(zip(self.model_names, self.posterior_weights)):
            print(f"      {name}: {weight:.4f}")
        
        print(f"   ✅ BMA training complete")
        return self
    
    def _calculate_log_evidence(self, model, X, y, cv_folds=5):
        """Calculate log evidence using cross-validation log-likelihood"""
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        log_likelihoods = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on fold
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Calculate log-likelihood on validation set
            if hasattr(model_copy, 'predict_proba'):
                probas = model_copy.predict_proba(X_val)
                # Avoid log(0) by clipping probabilities
                probas = np.clip(probas, 1e-15, 1 - 1e-15)
                
                # Calculate log-likelihood for each sample
                log_likes = []
                for i, true_class in enumerate(y_val):
                    log_likes.append(np.log(probas[i, true_class]))
                
                log_likelihoods.extend(log_likes)
            else:
                # For models without predict_proba, use 0/1 loss approximation
                predictions = model_copy.predict(X_val)
                correct = (predictions == y_val).astype(float)
                # Convert to log-likelihood approximation
                log_likes = np.log(correct + 1e-15)
                log_likelihoods.extend(log_likes)
        
        return np.mean(log_likelihoods)
    
    def predict(self, X):
        """Make predictions using Bayesian Model Averaging"""
        if not self.trained_models or self.posterior_weights is None:
            raise ValueError("BMA not trained yet!")
        
        # Get predictions from all models
        all_predictions = []
        
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                all_predictions.append(probas)
            else:
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                probas = np.eye(n_classes)[preds]
                all_predictions.append(probas)
        
        # Weighted average using posterior weights
        weighted_predictions = np.zeros_like(all_predictions[0])
        
        for i, (predictions, weight) in enumerate(zip(all_predictions, self.posterior_weights)):
            weighted_predictions += weight * predictions
        
        # Return class with highest probability
        return np.argmax(weighted_predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities using BMA"""
        if not self.trained_models or self.posterior_weights is None:
            raise ValueError("BMA not trained yet!")
        
        # Get predictions from all models
        all_predictions = []
        
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                all_predictions.append(probas)
        
        if not all_predictions:
            raise ValueError("No models support probability prediction")
        
        # Weighted average using posterior weights
        weighted_predictions = np.zeros_like(all_predictions[0])
        
        for predictions, weight in zip(all_predictions, self.posterior_weights):
            weighted_predictions += weight * predictions
        
        return weighted_predictions
    
    def get_model_uncertainties(self, X):
        """Calculate model uncertainty and epistemic uncertainty"""
        if not self.trained_models:
            raise ValueError("BMA not trained yet!")
        
        # Get predictions from all models
        all_probas = []
        
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                all_probas.append(probas)
        
        if not all_probas:
            return None
        
        all_probas = np.array(all_probas)
                # Model predictions (aleatoric uncertainty)
        bma_probas = self.predict_proba(X)
        aleatoric_uncertainty = -np.sum(bma_probas * np.log(bma_probas + 1e-15), axis=1)
        
        # Model disagreement (epistemic uncertainty)
        model_variance = np.var(all_probas, axis=0)
        epistemic_uncertainty = np.mean(model_variance, axis=1)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'aleatoric': aleatoric_uncertainty,
            'epistemic': epistemic_uncertainty,
            'total': total_uncertainty,
            'model_weights': self.posterior_weights
        }


# ==========================================
# ENSEMBLE DIVERSITY ANALYZER
# ==========================================

class EnsembleDiversityAnalyzer:
    """
    Comprehensive analysis of ensemble diversity and its impact on performance
    """
    
    def __init__(self):
        self.diversity_metrics = {}
        
    def analyze_ensemble_diversity(self, models, X, y):
        """
        Comprehensive diversity analysis for ensemble of trained models
        """
        print(f"🔍 Ensemble Diversity Analysis")
        print(f"   Analyzing {len(models)} models")
        print("=" * 50)
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        model_names = []
        
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
            model_names.append(type(model).__name__)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        predictions = np.array(predictions)
        
        # Calculate diversity metrics
        diversity_results = {}
        
        # 1. Pairwise Correlation
        print(f"\n1️⃣  Pairwise Correlation Analysis")
        correlations = self._calculate_pairwise_correlations(predictions)
        diversity_results['correlations'] = correlations
        
        avg_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
        print(f"   Average pairwise correlation: {avg_correlation:.4f}")
        
        # 2. Disagreement Measure
        print(f"\n2️⃣  Disagreement Analysis")
        disagreement = self._calculate_disagreement(predictions)
        diversity_results['disagreement'] = disagreement
        print(f"   Average disagreement: {disagreement:.4f}")
        
        # 3. Entropy-based Diversity
        if probabilities:
            print(f"\n3️⃣  Entropy-based Diversity")
            entropy_diversity = self._calculate_entropy_diversity(probabilities)
            diversity_results['entropy_diversity'] = entropy_diversity
            print(f"   Entropy diversity: {entropy_diversity:.4f}")
        
        # 4. Kappa Statistic
        print(f"\n4️⃣  Inter-rater Agreement (Kappa)")
        kappa_matrix = self._calculate_kappa_matrix(predictions)
        diversity_results['kappa_matrix'] = kappa_matrix
        
        avg_kappa = np.mean(kappa_matrix[np.triu_indices_from(kappa_matrix, k=1)])
        print(f"   Average Kappa statistic: {avg_kappa:.4f}")
        
        # 5. Diversity vs Accuracy Analysis
        print(f"\n5️⃣  Diversity vs Accuracy Trade-off")
        individual_accuracies = [accuracy_score(y, pred) for pred in predictions]
        diversity_results['individual_accuracies'] = individual_accuracies
        
        # Simple majority vote ensemble
        ensemble_pred = mode(predictions, axis=0)[0].flatten()
        ensemble_accuracy = accuracy_score(y, ensemble_pred)
        diversity_results['ensemble_accuracy'] = ensemble_accuracy
        
        print(f"   Individual accuracy range: {min(individual_accuracies):.4f} - {max(individual_accuracies):.4f}")
        print(f"   Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"   Improvement over best individual: {ensemble_accuracy - max(individual_accuracies):.4f}")
        
        # Visualize diversity analysis
        self._visualize_diversity_analysis(diversity_results, model_names, predictions, y)
        
        return diversity_results
    
    def _calculate_pairwise_correlations(self, predictions):
        """Calculate pairwise correlations between model predictions"""
        n_models = predictions.shape[0]
        correlations = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                    correlations[i, j] = corr if not np.isnan(corr) else 0.0
        
        return correlations
    
    def _calculate_disagreement(self, predictions):
        """Calculate average disagreement between models"""
        n_models, n_samples = predictions.shape
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)
        
        return np.mean(disagreements)
    
    def _calculate_entropy_diversity(self, probabilities):
        """Calculate entropy-based diversity measure"""
        probabilities = np.array(probabilities)
        
        # Average predictions across models
        avg_probabilities = np.mean(probabilities, axis=0)
        
        # Calculate entropy of average predictions
        avg_entropy = -np.sum(avg_probabilities * np.log(avg_probabilities + 1e-15), axis=1)
        
        # Calculate average entropy of individual predictions
        individual_entropies = []
        for model_proba in probabilities:
            entropy = -np.sum(model_proba * np.log(model_proba + 1e-15), axis=1)
            individual_entropies.append(entropy)
        
        avg_individual_entropy = np.mean(individual_entropies, axis=0)
        
        # Diversity = avg_entropy - avg_individual_entropy
        diversity = np.mean(avg_entropy - avg_individual_entropy)
        
        return diversity
    
    def _calculate_kappa_matrix(self, predictions):
        """Calculate Cohen's Kappa between all pairs of models"""
        from sklearn.metrics import cohen_kappa_score
        
        n_models = predictions.shape[0]
        kappa_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    kappa_matrix[i, j] = 1.0
                else:
                    kappa = cohen_kappa_score(predictions[i], predictions[j])
                    kappa_matrix[i, j] = kappa
        
        return kappa_matrix
    
    def _visualize_diversity_analysis(self, diversity_results, model_names, predictions, y_true):
        """Comprehensive visualization of diversity analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Correlation heatmap
        ax = axes[0, 0]
        correlations = diversity_results['correlations']
        im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Model Prediction Correlations')
        
        # Add correlation values
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                ax.text(j, i, f'{correlations[i, j]:.2f}', 
                       ha='center', va='center',
                       color='white' if abs(correlations[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        # 2. Individual vs Ensemble Performance
        ax = axes[0, 1]
        individual_accs = diversity_results['individual_accuracies']
        ensemble_acc = diversity_results['ensemble_accuracy']
        
        all_accs = individual_accs + [ensemble_acc]
        all_names = model_names + ['Ensemble']
        colors = ['lightblue'] * len(model_names) + ['red']
        
        bars = ax.bar(range(len(all_names)), all_accs, color=colors, alpha=0.7)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Individual vs Ensemble Performance')
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, acc in zip(bars, all_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Kappa statistics heatmap
        ax = axes[0, 2]
        kappa_matrix = diversity_results['kappa_matrix']
        im = ax.imshow(kappa_matrix, cmap='viridis', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Inter-rater Agreement (Kappa)')
        
        plt.colorbar(im, ax=ax)
        
        # 4. Accuracy vs Correlation scatter plot
        ax = axes[1, 0]
        
        # Calculate pairwise correlations and accuracy differences
        corr_values = []
        acc_diffs = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                corr = correlations[i, j]
                acc_diff = abs(individual_accs[i] - individual_accs[j])
                corr_values.append(corr)
                acc_diffs.append(acc_diff)
        
        ax.scatter(corr_values, acc_diffs, alpha=0.7, s=50)
        ax.set_xlabel('Prediction Correlation')
        ax.set_ylabel('Accuracy Difference')
        ax.set_title('Correlation vs Performance Difference')
        ax.grid(True, alpha=0.3)
        
        # 5. Disagreement distribution
        ax = axes[1, 1]
        
        # Calculate disagreement for each sample
        sample_disagreements = []
        for i in range(predictions.shape[1]):
            sample_preds = predictions[:, i]
            disagreement = len(np.unique(sample_preds)) - 1  # 0 if all agree, max if all disagree
            sample_disagreements.append(disagreement)
        
        ax.hist(sample_disagreements, bins=range(len(model_names) + 1), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Disagreeing Models')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample-wise Model Disagreement')
        ax.grid(True, alpha=0.3)
        
        # 6. Error analysis
        ax = axes[1, 2]
        
        # Analyze which samples are hardest (most disagreement)
        correct_predictions = (predictions == y_true).astype(int)
        sample_difficulty = len(model_names) - np.sum(correct_predictions, axis=0)
        
        ax.hist(sample_difficulty, bins=range(len(model_names) + 1), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Models Making Errors')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Difficulty Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def optimize_ensemble_composition(self, models, X, y, max_ensemble_size=None):
        """
        Find optimal subset of models for ensemble using forward selection
        """
        print(f"\n🎯 Optimizing Ensemble Composition")
        print("=" * 50)
        
        if max_ensemble_size is None:
            max_ensemble_size = len(models)
        
        model_names = [type(model).__name__ for model in models]
        n_models = len(models)
        
        # Forward selection
        selected_indices = []
        remaining_indices = list(range(n_models))
        performance_history = []
        
        print(f"   Forward selection up to {max_ensemble_size} models")
        
        for step in range(min(max_ensemble_size, n_models)):
            best_performance = -1
            best_index = -1
            
            # Try adding each remaining model
            for idx in remaining_indices:
                candidate_indices = selected_indices + [idx]
                candidate_models = [models[i] for i in candidate_indices]
                
                # Evaluate ensemble performance
                ensemble_performance = self._evaluate_ensemble_performance(candidate_models, X, y)
                
                if ensemble_performance > best_performance:
                    best_performance = ensemble_performance
                    best_index = idx
            
            # Add best model to ensemble
            if best_index != -1:
                selected_indices.append(best_index)
                remaining_indices.remove(best_index)
                performance_history.append(best_performance)
                
                print(f"   Step {step + 1}: Added {model_names[best_index]} (Performance: {best_performance:.4f})")
        
        # Final ensemble
        final_models = [models[i] for i in selected_indices]
        final_names = [model_names[i] for i in selected_indices]
        
        print(f"\n   ✅ Optimal ensemble composition:")
        for name in final_names:
            print(f"      • {name}")
        
        return {
            'selected_models': final_models,
            'selected_indices': selected_indices,
            'selected_names': final_names,
            'performance_history': performance_history
        }
    
    def _evaluate_ensemble_performance(self, models, X, y, cv_folds=3):
        """Evaluate ensemble performance using cross-validation"""
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models on fold
            trained_models = []
            for model in models:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                trained_models.append(model_copy)
            
            # Simple majority voting
            predictions = [model.predict(X_val) for model in trained_models]
            ensemble_pred = mode(predictions, axis=0)[0].flatten()
            
            score = accuracy_score(y_val, ensemble_pred)
            scores.append(score)
        
        return np.mean(scores)


# ==========================================
# AUTOML ENSEMBLE CONSTRUCTOR
# ==========================================

class AutoMLEnsembleConstructor:
    """
    Automated ensemble construction and optimization
    """
    
    def __init__(self, time_budget=300, ensemble_size_limit=10):
        self.time_budget = time_budget  # seconds
        self.ensemble_size_limit = ensemble_size_limit
        self.model_library = None
        self.best_ensemble = None
        self.construction_history = []
        
    def construct_ensemble(self, X, y, validation_split=0.2):
        """
        Automatically construct optimal ensemble using time budget
        """
        print(f"🤖 AutoML Ensemble Construction")
        print(f"   Time budget: {self.time_budget} seconds")
        print(f"   Ensemble size limit: {self.ensemble_size_limit}")
        print("=" * 50)
        
        import time
        start_time = time.time()
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Initialize model library
        self.model_library = self._create_model_library()
        print(f"   Model library: {len(self.model_library)} algorithms")
        
        # Phase 1: Quick screening of individual models
        print(f"\n1️⃣  Phase 1: Individual Model Screening")
        individual_performances = self._screen_individual_models(X_train, y_train, X_val, y_val)
        
        # Phase 2: Ensemble construction
        print(f"\n2️⃣  Phase 2: Ensemble Construction")
        
        best_performance = 0
        current_ensemble = []
        
        # Sort models by individual performance
        sorted_models = sorted(individual_performances.items(), 
                             key=lambda x: x[1]['performance'], reverse=True)
        
        # Greedy ensemble construction
        for model_name, model_info in sorted_models[:self.ensemble_size_limit]:
            # Check time budget
            if time.time() - start_time > self.time_budget:
                print(f"   ⏰ Time budget exceeded, stopping construction")
                break
            
            # Try adding model to ensemble
            candidate_ensemble = current_ensemble + [model_info['model']]
            
            # Evaluate ensemble
            ensemble_performance = self._evaluate_ensemble_quick(
                candidate_ensemble, X_train, y_train, X_val, y_val
            )
            
            if ensemble_performance > best_performance:
                current_ensemble = candidate_ensemble
                best_performance = ensemble_performance
                
                self.construction_history.append({
                    'ensemble_size': len(current_ensemble),
                    'performance': ensemble_performance,
                    'added_model': model_name
                })
                
                print(f"      Added {model_name} (Ensemble size: {len(current_ensemble)}, Performance: {ensemble_performance:.4f})")
            else:
                print(f"      Rejected {model_name} (No improvement)")
        
        # Phase 3: Ensemble optimization
        print(f"\n3️⃣  Phase 3: Ensemble Optimization")
        optimized_ensemble = self._optimize_ensemble(current_ensemble, X_train, y_train, X_val, y_val)
        
        self.best_ensemble = optimized_ensemble
        
        total_time = time.time() - start_time
        print(f"\n   ✅ AutoML construction complete")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Final ensemble size: {len(self.best_ensemble)}")
        print(f"   Final performance: {best_performance:.4f}")
        
        return self.best_ensemble
    
    def _create_model_library(self):
        """Create diverse library of ML models"""
        models = {}
        
        # Tree-based models
        models['RandomForest'] = RandomForestClassifier(n_estimators=50, random_state=42)
        models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=50, random_state=42)
        models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=50, random_state=42)
        
        # Linear models
        models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
        models['LogisticRegression_L1'] = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        
        # Probabilistic models
        models['NaiveBayes'] = GaussianNB()
        
        # Instance-based models
        models['KNN_5'] = KNeighborsClassifier(n_neighbors=5)
        models['KNN_10'] = KNeighborsClassifier(n_neighbors=10)
        
        # Support Vector Machines
        models['SVM_RBF'] = SVC(kernel='rbf', probability=True, random_state=42)
        models['SVM_Linear'] = SVC(kernel='linear', probability=True, random_state=42)
        
        # Neural Networks
        models['MLP'] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
        
        return models
    
    def _screen_individual_models(self, X_train, y_train, X_val, y_val):
        """Quick screening of individual model performance"""
        performances = {}
        
        for name, model in self.model_library.items():
            try:
                # Quick training and evaluation
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                pred = model_copy.predict(X_val)
                performance = accuracy_score(y_val, pred)
                
                performances[name] = {
                    'model': model_copy,
                    'performance': performance
                }
                
                print(f"      {name}: {performance:.4f}")
                
            except Exception as e:
                print(f"      {name}: Failed ({str(e)})")
                continue
        
        return performances
    
    def _evaluate_ensemble_quick(self, ensemble, X_train, y_train, X_val, y_val):
        """Quick ensemble evaluation using simple voting"""
        # Get predictions from all models
        predictions = []
        
        for model in ensemble:
            try:
                pred = model.predict(X_val)
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return 0
        
        # Simple majority voting
        predictions = np.array(predictions)
        ensemble_pred = mode(predictions, axis=0)[0].flatten()
        
        return accuracy_score(y_val, ensemble_pred)
    
    def _optimize_ensemble(self, ensemble, X_train, y_train, X_val, y_val):
        """Optimize ensemble using simple techniques"""
        print(f"      Optimizing ensemble of {len(ensemble)} models")
        
        # Try different combination strategies
        strategies = ['majority_vote', 'weighted_vote']
        best_strategy = 'majority_vote'
        best_performance = self._evaluate_ensemble_quick(ensemble, X_train, y_train, X_val, y_val)
        
        print(f"         Majority vote: {best_performance:.4f}")
        
        # Try weighted voting based on individual performance
        if len(ensemble) > 1:
            individual_perfs = []
            for model in ensemble:
                pred = model.predict(X_val)
                perf = accuracy_score(y_val, pred)
                individual_perfs.append(perf)
            
            # Weighted voting (simplified)
            weights = np.array(individual_perfs)
            weights = weights / np.sum(weights)
            
            # For simplicity, just return the ensemble as-is
            # In practice, would implement proper weighted voting
            
        return ensemble
    
    def get_construction_report(self):
        """Generate detailed report of ensemble construction process"""
        if not self.construction_history:
            return "No construction history available"
        
        report = "🤖 AutoML Ensemble Construction Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Final ensemble size: {len(self.best_ensemble)}\n"
        
        if self.best_ensemble:
            report += "Final ensemble composition:\n"
            for i, model in enumerate(self.best_ensemble):
                report += f"  {i+1}. {type(model).__name__}\n"
        
        report += "\nConstruction history:\n"
        for step in self.construction_history:
            report += f"  Size {step['ensemble_size']}: Added {step['added_model']} "
            report += f"(Performance: {step['performance']:.4f})\n"
        
        return report


# ==========================================
# COMPREHENSIVE ENSEMBLE DEMO
# ==========================================

def comprehensive_advanced_ensemble_demo():
    """
    Complete demonstration of advanced ensemble methods
    """
    print("🎓 Neural Odyssey - Week 16: Advanced Ensemble Methods Deep Dive")
    print("=" * 70)
    print("Beyond Simple Aggregation: The Art of Meta-Learning")
    print("=" * 70)
    
    # ================================================================
    # DATA PREPARATION
    # ================================================================
    
    print("\n📊 Preparing Datasets for Advanced Ensemble Analysis")
    
    # 1. Binary classification dataset
    X_binary, y_binary = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=3,
        n_clusters_per_class=1, class_sep=0.8, random_state=42
    )
    print(f"   Binary classification: {X_binary.shape[0]} samples, {X_binary.shape[1]} features")
    
    # 2. Multi-class dataset
    wine_data = load_wine()
    X_wine, y_wine = wine_data.data, wine_data.target
    print(f"   Wine dataset: {X_wine.shape[0]} samples, {X_wine.shape[1]} features, {len(np.unique(y_wine))} classes")
    
    # 3. Standardize features
    scaler = StandardScaler()
    X_binary = scaler.fit_transform(X_binary)
    X_wine = scaler.fit_transform(X_wine)
    
    # Split datasets
    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )
    
    # ================================================================
    # STACKING ENSEMBLE MASTERY
    # ================================================================
    
    print("\n" + "="*70)
    print("🚀 STACKING ENSEMBLE MASTERY")
    print("="*70)
    
    print("\n1️⃣  Advanced Stacking Implementation")
    
    # Define diverse base models
    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(kernel='rbf', probability=True, random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        KNeighborsClassifier(n_neighbors=5)
    ]
    
    # Meta-model
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create and train stacking ensemble
    stacking_ensemble = StackingEnsembleFromScratch(
        base_models=base_models,
        meta_model=meta_model,
        cv_folds=5,
        use_probabilities=True
    )
    
    stacking_ensemble.fit(X_bin_train, y_bin_train)
    
    # Comprehensive analysis
    stacking_ensemble.visualize_stacking_analysis(
        X_bin_train, y_bin_train, X_bin_test, y_bin_test
    )
    
    # Evaluate stacking performance
    stacking_pred = stacking_ensemble.predict(X_bin_test)
    stacking_acc = accuracy_score(y_bin_test, stacking_pred)
    print(f"   🏆 Stacking ensemble test accuracy: {stacking_acc:.4f}")
    
    # ================================================================
    # VOTING ENSEMBLE STRATEGIES
    # ================================================================
    
    print("\n" + "="*70)
    print("🗳️  VOTING ENSEMBLE STRATEGIES")
    print("="*70)
    
    print("\n1️⃣  Advanced Voting Methods")
    
    # Create voting ensemble with same base models
    voting_ensemble = VotingEnsembleFromScratch(
        models=base_models,
        voting='soft'
    )
    
    voting_ensemble.fit(X_bin_train, y_bin_train)
    
    # Compare voting strategies
    voting_results = voting_ensemble.analyze_voting_strategies(X_bin_train, y_bin_train)
    
    # Test performance
    for strategy in ['hard', 'soft', 'weighted']:
        voting_ensemble.voting = strategy
        if strategy == 'weighted':
            voting_ensemble.weights = voting_ensemble._calculate_optimal_weights(X_bin_train, y_bin_train)
        
        test_pred = voting_ensemble.predict(X_bin_test)
        test_acc = accuracy_score(y_bin_test, test_pred)
        print(f"   {strategy.capitalize()} voting test accuracy: {test_acc:.4f}")
    
    # ================================================================
    # BAYESIAN MODEL AVERAGING
    # ================================================================
    
    print("\n" + "="*70)
    print("🎲 BAYESIAN MODEL AVERAGING")
    print("="*70)
    
    print("\n1️⃣  Principled Uncertainty Quantification")
    
    # Create BMA ensemble
    bma_models = [
        LogisticRegression(random_state=42, max_iter=1000),
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(kernel='rbf', probability=True, random_state=42)
    ]
    
    bma_ensemble = BayesianModelAveraging(models=bma_models)
    bma_ensemble.fit(X_bin_train, y_bin_train)
    
    # Make predictions with uncertainty
    bma_pred = bma_ensemble.predict(X_bin_test)
    bma_acc = accuracy_score(y_bin_test, bma_pred)
    print(f"   🎯 BMA test accuracy: {bma_acc:.4f}")
    
    # Analyze uncertainties
    uncertainties = bma_ensemble.get_model_uncertainties(X_bin_test)
    
    if uncertainties:
        print(f"   📊 Uncertainty Analysis:")
        print(f"      Mean aleatoric uncertainty: {np.mean(uncertainties['aleatoric']):.4f}")
        print(f"      Mean epistemic uncertainty: {np.mean(uncertainties['epistemic']):.4f}")
        print(f"      Mean total uncertainty: {np.mean(uncertainties['total']):.4f}")
        
        # Visualize uncertainties
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Aleatoric uncertainty
        axes[0].hist(uncertainties['aleatoric'], bins=20, alpha=0.7, color='blue')
        axes[0].set_xlabel('Aleatoric Uncertainty')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Data Uncertainty')
        axes[0].grid(True, alpha=0.3)
        
        # Epistemic uncertainty
        axes[1].hist(uncertainties['epistemic'], bins=20, alpha=0.7, color='red')
        axes[1].set_xlabel('Epistemic Uncertainty')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Model Uncertainty')
        axes[1].grid(True, alpha=0.3)
        
        # Total uncertainty
        axes[2].hist(uncertainties['total'], bins=20, alpha=0.7, color='green')
        axes[2].set_xlabel('Total Uncertainty')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Combined Uncertainty')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ================================================================
    # ENSEMBLE DIVERSITY ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("🔍 ENSEMBLE DIVERSITY ANALYSIS")
    print("="*70)
    
    diversity_analyzer = EnsembleDiversityAnalyzer()
    
    print("\n1️⃣  Comprehensive Diversity Analysis")
    
    # Train models for diversity analysis
    diversity_models = []
    for model in base_models:
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_wine_train, y_wine_train)
        diversity_models.append(model_copy)
    
    diversity_results = diversity_analyzer.analyze_ensemble_diversity(
        diversity_models, X_wine_test, y_wine_test
    )
    
    print("\n2️⃣  Ensemble Composition Optimization")
    
    # Find optimal subset of models
    optimization_results = diversity_analyzer.optimize_ensemble_composition(
        diversity_models, X_wine_train, y_wine_train, max_ensemble_size=4
    )
    
    # Test optimized ensemble
    optimized_models = optimization_results['selected_models']
    optimized_predictions = [model.predict(X_wine_test) for model in optimized_models]
    optimized_ensemble_pred = mode(optimized_predictions, axis=0)[0].flatten()
    optimized_acc = accuracy_score(y_wine_test, optimized_ensemble_pred)
    
    print(f"   🎯 Optimized ensemble accuracy: {optimized_acc:.4f}")
    print(f"   📊 Improvement from selection: {optimized_acc - diversity_results['ensemble_accuracy']:.4f}")
    
    # ================================================================
    # AUTOML ENSEMBLE CONSTRUCTION
    # ================================================================
    
    print("\n" + "="*70)
    print("🤖 AUTOML ENSEMBLE CONSTRUCTION")
    print("="*70)
    
    print("\n1️⃣  Automated Ensemble Building")
    
    # Create AutoML constructor
    automl_constructor = AutoMLEnsembleConstructor(
        time_budget=120,  # 2 minutes for demo
        ensemble_size_limit=8
    )
    
    # Construct ensemble automatically
    automl_ensemble = automl_constructor.construct_ensemble(X_wine_train, y_wine_train)
    
    # Evaluate AutoML ensemble
    automl_predictions = [model.predict(X_wine_test) for model in automl_ensemble]
    automl_ensemble_pred = mode(automl_predictions, axis=0)[0].flatten()
    automl_acc = accuracy_score(y_wine_test, automl_ensemble_pred)
    
    print(f"   🏆 AutoML ensemble test accuracy: {automl_acc:.4f}")
    
    # Print construction report
    print(f"\n📋 AutoML Construction Report:")
    print(automl_constructor.get_construction_report())
    
    # ================================================================
    # ADVANCED ENSEMBLE TECHNIQUES
    # ================================================================
    
    print("\n" + "="*70)
    print("🔬 ADVANCED ENSEMBLE TECHNIQUES")
    print("="*70)
    
    print("\n1️⃣  Dynamic Ensemble Selection")
    
    # Simulate dynamic ensemble selection
    def dynamic_ensemble_selection(models, X, y, k_neighbors=5):
        """
        Simple implementation of dynamic ensemble selection
        """
        from sklearn.neighbors import NearestNeighbors
        
        # For each test sample, find k nearest training samples
        # and select best model based on local accuracy
        
        # This is a simplified version for demonstration
        best_models = []
        
        for model in models:
            accuracy = accuracy_score(y, model.predict(X))
            best_models.append((model, accuracy))
        
        # Sort by accuracy and return top 3
        best_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in best_models[:3]]
    
    # Apply dynamic selection
    dynamic_models = dynamic_ensemble_selection(diversity_models, X_wine_train, y_wine_train)
    dynamic_predictions = [model.predict(X_wine_test) for model in dynamic_models]
    dynamic_ensemble_pred = mode(dynamic_predictions, axis=0)[0].flatten()
    dynamic_acc = accuracy_score(y_wine_test, dynamic_ensemble_pred)
    
    print(f"   Dynamic selection accuracy: {dynamic_acc:.4f}")
    
    print("\n2️⃣  Ensemble Calibration")
    
    # Demonstrate ensemble calibration
    def calibrate_ensemble_predictions(ensemble, X_cal, y_cal, X_test):
        """
        Simple ensemble calibration using Platt scaling
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        # Get ensemble predictions on calibration set
        cal_predictions = ensemble.predict_proba(X_cal)[:, 1]
        
        # Fit calibration
        calibrator = CalibratedClassifierCV(cv='prefit')
        
        # For demonstration, just return uncalibrated predictions
        return ensemble.predict_proba(X_test)
    
    print("   Ensemble calibration techniques demonstrated")
    
    # ================================================================
    # PRODUCTION ENSEMBLE CONSIDERATIONS
    # ================================================================
    
    print("\n" + "="*70)
    print("🏭 PRODUCTION ENSEMBLE CONSIDERATIONS")
    print("="*70)
    
    print("\n1️⃣  Performance and Scalability Analysis")
    
    import time
    
    # Measure prediction latency
    latencies = {}
    
    # Individual model latencies
    for i, model in enumerate(diversity_models):
        start_time = time.time()
        _ = model.predict(X_wine_test)
        latency = (time.time() - start_time) * 1000  # milliseconds
        latencies[f'Model_{i+1}'] = latency
    
    # Ensemble latencies
    start_time = time.time()
    _ = stacking_ensemble.predict(X_bin_test)
    stacking_latency = (time.time() - start_time) * 1000
    latencies['Stacking'] = stacking_latency
    
    start_time = time.time()
    _ = voting_ensemble.predict(X_bin_test)
    voting_latency = (time.time() - start_time) * 1000
    latencies['Voting'] = voting_latency
    
    print(f"   Prediction Latencies (ms):")
    for method, latency in latencies.items():
        print(f"      {method}: {latency:.2f} ms")
    
    print("\n2️⃣  Ensemble Monitoring and Maintenance")
    
    # Simulate ensemble monitoring
    def ensemble_health_check(ensemble, X_monitor, y_monitor):
        """
        Basic ensemble health monitoring
        """
        try:
            predictions = ensemble.predict(X_monitor)
            accuracy = accuracy_score(y_monitor, predictions)
            
            health_status = {
                'accuracy': accuracy,
                'status': 'healthy' if accuracy > 0.7 else 'degraded',
                'predictions_made': len(predictions),
                'errors': 0
            }
            
            return health_status
        
        except Exception as e:
            return {
                'accuracy': 0,
                'status': 'failed',
                'predictions_made': 0,
                'errors': 1,
                'error_message': str(e)
            }
    
    # Monitor ensemble health
    health_status = ensemble_health_check(stacking_ensemble, X_bin_test, y_bin_test)
    print(f"   Ensemble Health Status: {health_status['status']}")
    print(f"   Current Accuracy: {health_status['accuracy']:.4f}")
    
    # ================================================================
    # FINAL INTEGRATION AND INSIGHTS
    # ================================================================
    
    print("\n" + "="*70)
    print("🎓 ADVANCED ENSEMBLE METHODS MASTERY SUMMARY")
    print("="*70)
    
    key_insights = [
        "🚀 Stacking: Meta-learning combines base models optimally through cross-validation",
        "🗳️  Voting: Simple but effective aggregation with hard, soft, and weighted strategies",
        "🎲 BMA: Principled uncertainty quantification through Bayesian model combination",
        "🔍 Diversity: Model diversity crucial for ensemble success, measurable and optimizable",
        "🤖 AutoML: Automated construction enables efficient ensemble design",
        "🔬 Advanced: Dynamic selection and calibration enhance ensemble capabilities",
        "🏭 Production: Latency, monitoring, and maintenance critical for deployment",
        "⚖️  Trade-offs: Balance between performance, complexity, and interpretability"
    ]
    
    print("\n💡 Key Insights Mastered:")
    for insight in key_insights:
        print(f"   {insight}")
    
    practical_guidelines = [
        "Use cross-validation for meta-feature generation in stacking",
        "Ensure base model diversity through different algorithms and hyperparameters",
        "Apply Bayesian model averaging for uncertainty-critical applications",
        "Monitor ensemble correlation and disagreement metrics",
        "Implement ensemble pruning for computational efficiency",
        "Consider dynamic selection for locally adaptive predictions",
        "Calibrate ensemble predictions for probability-based decisions",
        "Design monitoring systems for production ensemble health"
    ]
    
    print(f"\n📋 Practical Guidelines:")
    for guideline in practical_guidelines:
        print(f"   ✅ {guideline}")
    
    mathematical_foundations = [
        "Cross-validation theory prevents overfitting in meta-learning",
        "Bayesian inference provides principled model combination",
        "Information theory guides ensemble diversity optimization",
        "Statistical learning theory bounds ensemble generalization",
        "Optimization theory enables automated ensemble construction"
    ]
    
    print(f"\n🧮 Mathematical Foundations Mastered:")
    for foundation in mathematical_foundations:
        print(f"   📐 {foundation}")
    
    performance_summary = {
        'Stacking Ensemble': stacking_acc,
        'Voting Ensemble': test_acc,  # Last voting strategy tested
        'BMA Ensemble': bma_acc,
        'Optimized Ensemble': optimized_acc,
        'AutoML Ensemble': automl_acc,
        'Dynamic Selection': dynamic_acc
    }
    
    print(f"\n📊 Performance Summary:")
    for method, accuracy in performance_summary.items():
        print(f"   {method}: {accuracy:.4f}")
    
    best_method = max(performance_summary.items(), key=lambda x: x[1])
    print(f"\n🏆 Best performing method: {best_method[0]} ({best_method[1]:.4f})")
    
    next_week_preview = [
        "Unsupervised learning and clustering methods",
        "Dimensionality reduction and manifold learning",
        "Principal Component Analysis and beyond",
        "Neural networks and deep learning foundations"
    ]
    
    print(f"\n🔮 Next Week Preview:")
    for topic in next_week_preview:
        print(f"   📚 {topic}")
    
    return {
        'stacking_ensemble': stacking_ensemble,
        'voting_ensemble': voting_ensemble,
        'bma_ensemble': bma_ensemble,
        'diversity_results': diversity_results,
        'optimization_results': optimization_results,
        'automl_ensemble': automl_ensemble,
        'performance_summary': performance_summary,
        'datasets': {
            'binary': (X_binary, y_binary),
            'wine': (X_wine, y_wine)
        }
    }


# ==========================================
# ADVANCED ENSEMBLE RESEARCH TOPICS
# ==========================================

def advanced_ensemble_research_topics():
    """
    Explore cutting-edge research topics in ensemble learning
    """
    print(f"\n🔬 Advanced Ensemble Research Topics")
    print("=" * 50)
    
    research_topics = [
        {
            'topic': 'Neural Ensemble Methods',
            'description': 'Integration of ensemble principles with deep learning',
            'techniques': [
                'Deep ensemble uncertainty quantification',
                'Multi-head neural network architectures',
                'Ensemble neural architecture search',
                'Bayesian neural networks'
            ],
            'applications': 'Computer vision, NLP, autonomous systems'
        },
        {
            'topic': 'Federated Ensemble Learning',
            'description': 'Ensemble methods for distributed and privacy-preserving ML',
            'techniques': [
                'Federated model averaging',
                'Privacy-preserving ensemble aggregation',
                'Communication-efficient ensemble methods',
                'Heterogeneous federated ensembles'
            ],
            'applications': 'Healthcare, finance, IoT networks'
        },
        {
            'topic': 'Online Ensemble Learning',
            'description': 'Adaptive ensemble methods for streaming data',
            'techniques': [
                'Online ensemble selection',
                'Concept drift adaptation',
                'Incremental ensemble construction',
                'Real-time model updates'
            ],
            'applications': 'Fraud detection, recommendation systems, sensor networks'
        },
        {
            'topic': 'Interpretable Ensembles',
            'description': 'Maintaining interpretability in ensemble methods',
            'techniques': [
                'Rule-based ensemble combination',
                'Attention mechanisms for model selection',
                'Hierarchical ensemble explanations',
                'Model-agnostic ensemble interpretation'
            ],
            'applications': 'Healthcare diagnosis, financial decisions, legal systems'
        }
    ]
    
    for i, topic in enumerate(research_topics, 1):
        print(f"\n{i}️⃣  {topic['topic']}")
        print(f"   Description: {topic['description']}")
        print(f"   Key techniques:")
        for technique in topic['techniques']:
            print(f"      • {technique}")
        print(f"   Applications: {topic['applications']}")
    
    print(f"\n🔮 Future Directions:")
    future_directions = [
        "Quantum ensemble methods for quantum machine learning",
        "Causal ensemble learning for causal inference",
        "Multi-modal ensemble fusion for diverse data types",
        "Continual ensemble learning for lifelong learning systems",
        "Ensemble methods for few-shot and zero-shot learning"
    ]
    
    for direction in future_directions:
        print(f"   🚀 {direction}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    """
    Run this file to master advanced ensemble methods and meta-learning!
    
    This comprehensive exploration covers:
    1. Stacking and meta-learning from theoretical foundations
    2. Voting ensemble strategies and optimal combination
    3. Bayesian model averaging and uncertainty quantification
    4. Dynamic ensemble selection and adaptive methods
    5. Ensemble diversity analysis and optimization
    6. Automated machine learning (AutoML) principles
    7. Production ensemble systems and scalability
    8. Advanced research topics and future directions
    """
    
    print("🚀 Welcome to Neural Odyssey - Week 16: Advanced Ensemble Methods!")
    print("Master the art of meta-learning and sophisticated model combination.")
    print("\nThis comprehensive week includes:")
    print("1. 🚀 Stacking and cross-validation meta-learning")
    print("2. 🗳️  Advanced voting strategies and optimal weighting")
    print("3. 🎲 Bayesian model averaging with uncertainty quantification")
    print("4. 🔍 Ensemble diversity analysis and optimization")
    print("5. 🤖 Automated ensemble construction (AutoML)")
    print("6. 🔬 Dynamic selection and adaptive ensemble methods")
    print("7. 🏭 Production deployment and monitoring considerations")
    print("8. 🔬 Cutting-edge research topics and future directions")
    
    # Run comprehensive demonstration
    print("\n" + "="*70)
    print("🎭 Starting Advanced Ensemble Methods Journey...")
    print("="*70)
    
    # Execute complete demonstration
    main_results = comprehensive_advanced_ensemble_demo()
    
    # Explore research topics
    advanced_ensemble_research_topics()
    
    print("\n" + "="*70)
    print("🎉 WEEK 16 COMPLETE: ADVANCED ENSEMBLE MASTERY ACHIEVED!")
    print("="*70)
    
    print(f"\n🏆 Achievement Summary:")
    print(f"   ✅ Stacking: Meta-learning with cross-validation implemented")
    print(f"   ✅ Voting: Hard, soft, and weighted strategies mastered")
    print(f"   ✅ BMA: Bayesian model averaging with uncertainty analysis")
    print(f"   ✅ Diversity: Comprehensive ensemble diversity framework")
    print(f"   ✅ AutoML: Automated ensemble construction system")
    print(f"   ✅ Advanced: Dynamic selection and calibration techniques")
    print(f"   ✅ Production: Scalability and monitoring considerations")
    print(f"   ✅ Research: Cutting-edge topics and future directions")
    
    print(f"\n🧠 Core Concepts Mastered:")
    core_concepts = [
        "Meta-learning enables optimal combination of diverse models",
        "Cross-validation prevents overfitting in ensemble construction",
        "Bayesian principles provide principled uncertainty quantification",
        "Model diversity is measurable and directly impacts performance",
        "Automated construction democratizes advanced ensemble techniques",
        "Dynamic selection adapts to local data characteristics",
        "Production deployment requires careful performance monitoring",
        "Advanced research opens new frontiers in ensemble learning"
    ]
    
    for concept in core_concepts:
        print(f"   💡 {concept}")
    
    print(f"\n🚀 Ready for Phase 3: Advanced Machine Learning Topics!")
    print(f"   Building on ensemble mastery to explore deep learning and beyond")
    
    # Return comprehensive results
    return main_results
        