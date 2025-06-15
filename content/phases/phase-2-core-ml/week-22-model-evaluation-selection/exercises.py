"""
Neural Learning Web - Phase 2: Week 22 - Model Evaluation and Selection
=======================================================================

Week 22: Model Evaluation and Selection
Master the art of evaluating and selecting the best machine learning models
through comprehensive validation techniques, performance metrics, and 
statistical significance testing.

This week bridges foundational concepts with advanced model selection strategies,
focusing on robust evaluation frameworks that prevent overfitting and ensure
reliable model performance in production environments.

Learning Objectives:
- Master comprehensive model evaluation methodologies
- Implement robust cross-validation strategies  
- Understand statistical significance testing for model comparison
- Build automated model selection pipelines
- Develop performance monitoring and model drift detection systems
- Create interpretable evaluation frameworks for stakeholder communication

Author: Neural Explorer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_wine
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,
                                   StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit,
                                   GridSearchCV, RandomizedSearchCV, validation_curve,
                                   learning_curve, permutation_test_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, precision_recall_curve, roc_curve,
                           confusion_matrix, classification_report, log_loss,
                           mean_squared_error, mean_absolute_error, r2_score,
                           explained_variance_score, max_error)
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# COMPREHENSIVE MODEL EVALUATION FRAMEWORK
# ==========================================

class ModelEvaluationFramework:
    """
    Complete framework for evaluating and comparing machine learning models
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.evaluation_results = {}
        self.model_performances = {}
        
    def evaluate_single_model(self, model, X, y, cv_folds=5, scoring_metrics=None):
        """
        Comprehensive evaluation of a single model
        """
        if scoring_metrics is None:
            if self.problem_type == 'classification':
                scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 
                                 'f1_macro', 'roc_auc_ovr']
            else:
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 
                                 'r2', 'explained_variance']
        
        print(f"🔍 Evaluating {type(model).__name__}")
        print(f"   Problem Type: {self.problem_type}")
        print(f"   CV Folds: {cv_folds}")
        print(f"   Metrics: {scoring_metrics}")
        
        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            model, X, y, cv=cv_folds, scoring=scoring_metrics,
            return_train_score=True, return_estimator=True, n_jobs=-1
        )
        
        # Organize results
        results = {
            'cv_results': cv_results,
            'fitted_models': cv_results['estimator'],
            'metrics_summary': {}
        }
        
        # Calculate summary statistics for each metric
        for metric in scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['metrics_summary'][metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_scores': test_scores,
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_scores': train_scores,
                'overfitting': np.mean(train_scores) - np.mean(test_scores)
            }
        
        print(f"   ✅ Evaluation complete")
        self._display_single_model_results(results, type(model).__name__)
        
        return results
    
    def _display_single_model_results(self, results, model_name):
        """Display formatted results for single model evaluation"""
        print(f"\n📊 {model_name} Performance Summary:")
        print("=" * 60)
        
        for metric, stats in results['metrics_summary'].items():
            print(f"{metric.upper()}:")
            print(f"  Test:  {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
            print(f"  Train: {stats['train_mean']:.4f} ± {stats['train_std']:.4f}")
            print(f"  Gap:   {stats['overfitting']:.4f}")
            print()
    
    def compare_multiple_models(self, models_dict, X, y, cv_folds=5, scoring_metrics=None):
        """
        Compare multiple models using cross-validation
        """
        print(f"🏆 Multi-Model Comparison")
        print(f"   Models: {list(models_dict.keys())}")
        print(f"   Dataset shape: {X.shape}")
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            print(f"\n   Evaluating {model_name}...")
            results = self.evaluate_single_model(model, X, y, cv_folds, scoring_metrics)
            comparison_results[model_name] = results
        
        # Store for statistical testing
        self.model_performances = comparison_results
        
        # Create comparison visualization
        self._visualize_model_comparison(comparison_results)
        
        # Recommend best model
        primary_metric = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        best_model = self._get_best_model(comparison_results, primary_metric)
        
        print(f"\n🏅 Best Model: {best_model}")
        
        return comparison_results
    
    def _get_best_model(self, results, metric):
        """Identify best model based on specified metric"""
        if metric.startswith('neg_'):
            # For negative metrics (higher is better when negated)
            best_model = max(results.keys(), 
                           key=lambda k: results[k]['metrics_summary'][metric]['test_mean'])
        else:
            # For positive metrics (higher is better)
            best_model = max(results.keys(),
                           key=lambda k: results[k]['metrics_summary'][metric]['test_mean'])
        return best_model
    
    def _visualize_model_comparison(self, comparison_results):
        """Create comprehensive visualization of model comparison"""
        models = list(comparison_results.keys())
        n_models = len(models)
        
        # Get primary metric
        primary_metric = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Primary metric comparison
        ax = axes[0, 0]
        test_means = [comparison_results[m]['metrics_summary'][primary_metric]['test_mean'] 
                     for m in models]
        test_stds = [comparison_results[m]['metrics_summary'][primary_metric]['test_std'] 
                    for m in models]
        
        bars = ax.bar(range(n_models), test_means, yerr=test_stds, 
                     alpha=0.7, capsize=5, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel(primary_metric.replace('_', ' ').title())
        ax.set_title(f'{primary_metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, test_means, test_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Overfitting analysis
        ax = axes[0, 1]
        overfitting_gaps = [comparison_results[m]['metrics_summary'][primary_metric]['overfitting'] 
                           for m in models]
        
        colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' 
                 for gap in overfitting_gaps]
        bars = ax.bar(range(n_models), overfitting_gaps, alpha=0.7, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='Caution')
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Overfitting')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Train-Test Gap')
        ax.set_title('Overfitting Analysis')
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Performance distribution (box plots)
        ax = axes[1, 0]
        test_scores_list = [comparison_results[m]['metrics_summary'][primary_metric]['test_scores'] 
                           for m in models]
        
        bp = ax.boxplot(test_scores_list, labels=models, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel(primary_metric.replace('_', ' ').title())
        ax.set_title('Performance Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Multiple metrics radar chart (if classification)
        ax = axes[1, 1]
        if self.problem_type == 'classification' and len(comparison_results[models[0]]['metrics_summary']) > 1:
            self._create_radar_chart(ax, comparison_results, models)
        else:
            ax.text(0.5, 0.5, 'Radar Chart\n(Classification Only)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Multi-Metric Comparison')
        
        plt.tight_layout()
        plt.show()
    
    def _create_radar_chart(self, ax, comparison_results, models):
        """Create radar chart for multi-metric comparison"""
        metrics = list(comparison_results[models[0]]['metrics_summary'].keys())
        n_metrics = len(metrics)
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models[:3]):  # Limit to 3 models for clarity
            values = [comparison_results[model]['metrics_summary'][metric]['test_mean'] 
                     for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_title('Multi-Metric Performance')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)

# ==========================================
# STATISTICAL SIGNIFICANCE TESTING
# ==========================================

class StatisticalSignificanceTester:
    """
    Framework for statistical significance testing of model comparisons
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.test_results = {}
    
    def paired_ttest(self, scores1, scores2, model1_name, model2_name):
        """
        Perform paired t-test for comparing two models
        """
        print(f"🔬 Paired T-Test: {model1_name} vs {model2_name}")
        
        # Perform paired t-test
        statistic, p_value = ttest_rel(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        diff = scores1 - scores2
        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        cohens_d = np.mean(diff) / pooled_std
        
        # Interpret results
        is_significant = p_value < self.alpha
        winner = model1_name if np.mean(scores1) > np.mean(scores2) else model2_name
        
        results = {
            'test_type': 'paired_ttest',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'is_significant': is_significant,
            'winner': winner if is_significant else 'No significant difference',
            'effect_size': cohens_d,
            'mean_diff': np.mean(diff),
            'confidence_interval': self._calculate_ci(diff)
        }
        
        self._display_test_results(results, model1_name, model2_name)
        return results
    
    def wilcoxon_test(self, scores1, scores2, model1_name, model2_name):
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test)
        """
        print(f"🔬 Wilcoxon Signed-Rank Test: {model1_name} vs {model2_name}")
        
        statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
        
        is_significant = p_value < self.alpha
        winner = model1_name if np.median(scores1) > np.median(scores2) else model2_name
        
        results = {
            'test_type': 'wilcoxon',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'is_significant': is_significant,
            'winner': winner if is_significant else 'No significant difference',
            'median_diff': np.median(scores1) - np.median(scores2)
        }
        
        self._display_test_results(results, model1_name, model2_name)
        return results
    
    def friedman_test(self, scores_dict):
        """
        Perform Friedman test for comparing multiple models
        """
        print(f"🔬 Friedman Test: Comparing {len(scores_dict)} models")
        
        model_names = list(scores_dict.keys())
        scores_matrix = np.array([scores_dict[name] for name in model_names])
        
        statistic, p_value = friedmanchisquare(*scores_matrix)
        
        is_significant = p_value < self.alpha
        
        results = {
            'test_type': 'friedman',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'is_significant': is_significant,
            'models': model_names,
            'interpretation': 'At least one model differs significantly' if is_significant else 'No significant differences'
        }
        
        print(f"   Friedman statistic: {statistic:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Significant: {'Yes' if is_significant else 'No'}")
        print(f"   Interpretation: {results['interpretation']}")
        
        # If significant, perform post-hoc pairwise comparisons
        if is_significant:
            print(f"\n   Performing post-hoc pairwise comparisons...")
            pairwise_results = self._pairwise_comparisons(scores_dict)
            results['pairwise_comparisons'] = pairwise_results
        
        return results
    
    def _pairwise_comparisons(self, scores_dict):
        """Perform pairwise comparisons after significant Friedman test"""
        model_names = list(scores_dict.keys())
        pairwise_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Use Bonferroni correction for multiple comparisons
                adjusted_alpha = self.alpha / (len(model_names) * (len(model_names) - 1) / 2)
                
                result = self.wilcoxon_test(scores_dict[model1], scores_dict[model2], 
                                          model1, model2)
                result['adjusted_alpha'] = adjusted_alpha
                result['is_significant_adjusted'] = result['p_value'] < adjusted_alpha
                
                pairwise_results[f"{model1}_vs_{model2}"] = result
        
        return pairwise_results
    
    def _calculate_ci(self, diff, confidence=0.95):
        """Calculate confidence interval for difference"""
        alpha = 1 - confidence
        n = len(diff)
        se = stats.sem(diff)
        h = se * stats.t.ppf((1 + confidence) / 2, n-1)
        
        mean_diff = np.mean(diff)
        return (mean_diff - h, mean_diff + h)
    
    def _display_test_results(self, results, model1_name, model2_name):
        """Display formatted test results"""
        print(f"   Statistic: {results['statistic']:.4f}")
        print(f"   P-value: {results['p_value']:.6f}")
        print(f"   Significant: {'Yes' if results['is_significant'] else 'No'}")
        print(f"   Winner: {results['winner']}")
        
        if 'effect_size' in results:
            print(f"   Effect size (Cohen's d): {results['effect_size']:.4f}")
            effect_magnitude = 'Small' if abs(results['effect_size']) < 0.5 else 'Medium' if abs(results['effect_size']) < 0.8 else 'Large'
            print(f"   Effect magnitude: {effect_magnitude}")

# ==========================================
# ADVANCED CROSS-VALIDATION STRATEGIES
# ==========================================

class AdvancedCrossValidation:
    """
    Advanced cross-validation strategies for different scenarios
    """
    
    def __init__(self):
        self.cv_strategies = {}
    
    def compare_cv_strategies(self, X, y, model, strategies=None):
        """
        Compare different cross-validation strategies
        """
        if strategies is None:
            strategies = {
                'KFold_5': KFold(n_splits=5, shuffle=True, random_state=42),
                'KFold_10': KFold(n_splits=10, shuffle=True, random_state=42),
                'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                'StratifiedKFold_10': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            }
        
        print(f"🔄 Comparing Cross-Validation Strategies")
        print(f"   Model: {type(model).__name__}")
        print(f"   Strategies: {list(strategies.keys())}")
        
        results = {}
        
        for strategy_name, cv_strategy in strategies.items():
            print(f"\n   Testing {strategy_name}...")
            
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
            
            results[strategy_name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'cv_strategy': cv_strategy
            }
            
            print(f"      Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            print(f"      Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        # Visualize comparison
        self._visualize_cv_comparison(results)
        
        return results
    
    def _visualize_cv_comparison(self, results):
        """Visualize cross-validation strategy comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategies = list(results.keys())
        
        # Box plot of scores
        scores_list = [results[strategy]['scores'] for strategy in strategies]
        bp = ax1.boxplot(scores_list, labels=strategies, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('CV Strategy')
        ax1.set_ylabel('Accuracy Score')
        ax1.set_title('Score Distribution by CV Strategy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Mean and std comparison
        means = [results[strategy]['mean'] for strategy in strategies]
        stds = [results[strategy]['std'] for strategy in strategies]
        
        bars = ax2.bar(range(len(strategies)), means, yerr=stds, 
                      alpha=0.7, capsize=5, color='skyblue')
        ax2.set_xlabel('CV Strategy')
        ax2.set_ylabel('Mean Accuracy')
        ax2.set_title('Mean Performance by CV Strategy')
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def nested_cross_validation(self, X, y, model, param_grid, outer_cv=5, inner_cv=3):
        """
        Perform nested cross-validation for unbiased performance estimation
        """
        print(f"🎯 Nested Cross-Validation")
        print(f"   Model: {type(model).__name__}")
        print(f"   Outer CV: {outer_cv} folds")
        print(f"   Inner CV: {inner_cv} folds")
        print(f"   Parameter grid: {param_grid}")
        
        outer_cv_strategy = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        inner_cv_strategy = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        
        # Nested CV scores
        nested_scores = []
        best_params_list = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv_strategy.split(X, y)):
            print(f"   Outer fold {fold + 1}/{outer_cv}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv_strategy, 
                scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train_outer, y_train_outer)
            best_params_list.append(grid_search.best_params_)
            
            # Evaluate on outer test set
            test_score = grid_search.score(X_test_outer, y_test_outer)
            nested_scores.append(test_score)
            
            print(f"      Best params: {grid_search.best_params_}")
            print(f"      Test score: {test_score:.4f}")
        
        # Calculate final statistics
        nested_mean = np.mean(nested_scores)
        nested_std = np.std(nested_scores)
        
        print(f"\n   📊 Nested CV Results:")
        print(f"      Mean score: {nested_mean:.4f} ± {nested_std:.4f}")
        print(f"      Individual scores: {[f'{s:.4f}' for s in nested_scores]}")
        
        # Analyze parameter stability
        print(f"\n   🔧 Parameter Stability Analysis:")
        all_params = set()
        for params in best_params_list:
            all_params.update(params.keys())
        
        for param in all_params:
            values = [params.get(param, 'Not selected') for params in best_params_list]
            unique_values, counts = np.unique(values, return_counts=True)
            print(f"      {param}:")
            for value, count in zip(unique_values, counts):
                print(f"        {value}: {count}/{outer_cv} folds ({count/outer_cv:.1%})")
        
        return {
            'nested_scores': nested_scores,
            'nested_mean': nested_mean,
            'nested_std': nested_std,
            'best_params_per_fold': best_params_list
        }

# ==========================================
# AUTOMATED MODEL SELECTION PIPELINE
# ==========================================

class AutomatedModelSelection:
    """
    Automated pipeline for model selection and evaluation
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.model_library = self._create_model_library()
        self.evaluation_framework = ModelEvaluationFramework(problem_type)
        self.significance_tester = StatisticalSignificanceTester()
        
    def _create_model_library(self):
        """Create library of models to evaluate"""
        if self.problem_type == 'classification':
            models = {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVM_RBF': SVC(random_state=42, probability=True),
                'SVM_Linear': SVC(kernel='linear', random_state=42, probability=True),
                'KNN': KNeighborsClassifier(),
                'NaiveBayes': GaussianNB(),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'MLP': MLPClassifier(random_state=42, max_iter=1000)
            }
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import LinearRegression
            
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'SVR_RBF': SVR(),
                'SVR_Linear': SVR(kernel='linear')
            }
        
        return models
    
    def run_automated_selection(self, X, y, cv_folds=5, top_k=3):
        """
        Run automated model selection pipeline
        """
        print(f"🤖 Automated Model Selection Pipeline")
        print(f"   Problem Type: {self.problem_type}")
        print(f"   Dataset Shape: {X.shape}")
        print(f"   Models to evaluate: {len(self.model_library)}")
        print(f"   CV Folds: {cv_folds}")
        print(f"   Top models to analyze: {top_k}")
        
        # Step 1: Initial screening
        print(f"\n📋 Step 1: Initial Model Screening")
        comparison_results = self.evaluation_framework.compare_multiple_models(
            self.model_library, X, y, cv_folds
        )
        
        # Step 2: Select top models
        primary_metric = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        top_models = self._select_top_models(comparison_results, primary_metric, top_k)
        
        print(f"\n🏅 Step 2: Top {top_k} Models Selected")
        for i, (model_name, score) in enumerate(top_models, 1):
            print(f"   {i}. {model_name}: {score:.4f}")
        
        # Step 3: Statistical significance testing
        print(f"\n🔬 Step 3: Statistical Significance Testing")
        significance_results = self._test_top_models_significance(
            comparison_results, [model[0] for model in top_models], primary_metric
        )
        
        # Step 4: Final recommendation
        print(f"\n🎯 Step 4: Final Recommendation")
        final_recommendation = self._make_final_recommendation(
            top_models, significance_results, comparison_results
        )
        
        return {'comparison_results': comparison_results,
           'top_models': top_models,
           'significance_results': significance_results,
           'final_recommendation': final_recommendation
       }
   
   def _select_top_models(self, comparison_results, metric, top_k):
       """Select top k models based on performance metric"""
       model_scores = []
       
       for model_name, results in comparison_results.items():
           score = results['metrics_summary'][metric]['test_mean']
           model_scores.append((model_name, score))
       
       # Sort by score (descending for positive metrics, ascending for negative)
       if metric.startswith('neg_'):
           model_scores.sort(key=lambda x: x[1], reverse=True)  # Higher (less negative) is better
       else:
           model_scores.sort(key=lambda x: x[1], reverse=True)  # Higher is better
       
       return model_scores[:top_k]
   
   def _test_top_models_significance(self, comparison_results, top_model_names, metric):
       """Test statistical significance between top models"""
       significance_results = {}
       
       for i, model1 in enumerate(top_model_names):
           for model2 in top_model_names[i+1:]:
               scores1 = comparison_results[model1]['metrics_summary'][metric]['test_scores']
               scores2 = comparison_results[model2]['metrics_summary'][metric]['test_scores']
               
               # Perform both parametric and non-parametric tests
               ttest_result = self.significance_tester.paired_ttest(
                   scores1, scores2, model1, model2
               )
               wilcoxon_result = self.significance_tester.wilcoxon_test(
                   scores1, scores2, model1, model2
               )
               
               significance_results[f"{model1}_vs_{model2}"] = {
                   'ttest': ttest_result,
                   'wilcoxon': wilcoxon_result
               }
       
       return significance_results
   
   def _make_final_recommendation(self, top_models, significance_results, comparison_results):
       """Make final model recommendation based on performance and significance"""
       best_model_name = top_models[0][0]
       best_model_score = top_models[0][1]
       
       # Check if best model is significantly better than others
       significant_improvements = []
       
       for comparison_key, results in significance_results.items():
           if best_model_name in comparison_key:
               if (results['ttest']['is_significant'] and 
                   results['wilcoxon']['is_significant'] and
                   results['ttest']['winner'] == best_model_name):
                   
                   other_model = comparison_key.replace(f"{best_model_name}_vs_", "").replace(f"_vs_{best_model_name}", "")
                   significant_improvements.append(other_model)
       
       # Calculate additional metrics for recommendation
       best_model_results = comparison_results[best_model_name]['metrics_summary']
       primary_metric = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
       
       overfitting_score = best_model_results[primary_metric]['overfitting']
       stability_score = 1 / (best_model_results[primary_metric]['test_std'] + 1e-6)
       
       recommendation = {
           'recommended_model': best_model_name,
           'performance_score': best_model_score,
           'overfitting_score': overfitting_score,
           'stability_score': stability_score,
           'significantly_better_than': significant_improvements,
           'confidence_level': 'High' if len(significant_improvements) >= 2 else 'Medium' if len(significant_improvements) >= 1 else 'Low',
           'reasoning': self._generate_recommendation_reasoning(
               best_model_name, best_model_score, overfitting_score, 
               stability_score, significant_improvements
           )
       }
       
       print(f"   Recommended Model: {best_model_name}")
       print(f"   Performance Score: {best_model_score:.4f}")
       print(f"   Confidence Level: {recommendation['confidence_level']}")
       print(f"   Reasoning: {recommendation['reasoning']}")
       
       return recommendation
   
   def _generate_recommendation_reasoning(self, model_name, score, overfitting, 
                                        stability, significant_improvements):
       """Generate human-readable reasoning for recommendation"""
       reasons = []
       
       reasons.append(f"Highest performance score ({score:.4f})")
       
       if overfitting < 0.02:
           reasons.append("Low overfitting risk")
       elif overfitting > 0.05:
           reasons.append("Warning: High overfitting detected")
       
       if stability > 10:
           reasons.append("High stability across CV folds")
       elif stability < 5:
           reasons.append("Moderate stability concerns")
       
       if len(significant_improvements) > 0:
           reasons.append(f"Significantly outperforms {len(significant_improvements)} other models")
       
       return "; ".join(reasons)

# ==========================================
# PERFORMANCE MONITORING AND DRIFT DETECTION
# ==========================================

class ModelPerformanceMonitor:
   """
   Monitor model performance over time and detect concept drift
   """
   
   def __init__(self, baseline_performance, threshold=0.05):
       self.baseline_performance = baseline_performance
       self.threshold = threshold
       self.performance_history = []
       self.drift_alerts = []
   
   def update_performance(self, new_performance, timestamp=None):
       """Update performance history with new measurements"""
       if timestamp is None:
           timestamp = len(self.performance_history)
       
       self.performance_history.append({
           'timestamp': timestamp,
           'performance': new_performance,
           'drift_detected': False
       })
       
       # Check for drift
       drift_detected = self._detect_drift(new_performance)
       
       if drift_detected:
           self.performance_history[-1]['drift_detected'] = True
           self.drift_alerts.append({
               'timestamp': timestamp,
               'performance_drop': self.baseline_performance - new_performance,
               'severity': self._calculate_drift_severity(new_performance)
           })
           
           print(f"⚠️  Drift Alert at timestamp {timestamp}")
           print(f"   Performance drop: {self.baseline_performance - new_performance:.4f}")
           print(f"   Severity: {self._calculate_drift_severity(new_performance)}")
       
       return drift_detected
   
   def _detect_drift(self, current_performance):
       """Detect if current performance indicates concept drift"""
       performance_drop = self.baseline_performance - current_performance
       return performance_drop > self.threshold
   
   def _calculate_drift_severity(self, current_performance):
       """Calculate drift severity level"""
       performance_drop = self.baseline_performance - current_performance
       
       if performance_drop > 0.15:
           return 'Critical'
       elif performance_drop > 0.10:
           return 'High'
       elif performance_drop > 0.05:
           return 'Medium'
       else:
           return 'Low'
   
   def plot_performance_history(self):
       """Visualize performance history and drift alerts"""
       if not self.performance_history:
           print("No performance history to plot")
           return
       
       fig, ax = plt.subplots(figsize=(12, 6))
       
       timestamps = [entry['timestamp'] for entry in self.performance_history]
       performances = [entry['performance'] for entry in self.performance_history]
       drift_points = [entry['timestamp'] for entry in self.performance_history if entry['drift_detected']]
       drift_performances = [entry['performance'] for entry in self.performance_history if entry['drift_detected']]
       
       # Plot performance over time
       ax.plot(timestamps, performances, 'b-o', label='Performance', alpha=0.7)
       
       # Highlight drift points
       if drift_points:
           ax.scatter(drift_points, drift_performances, color='red', s=100, 
                     label='Drift Detected', zorder=5)
       
       # Add baseline and threshold lines
       ax.axhline(y=self.baseline_performance, color='green', linestyle='--', 
                 label='Baseline Performance', alpha=0.7)
       ax.axhline(y=self.baseline_performance - self.threshold, color='orange', 
                 linestyle='--', label='Drift Threshold', alpha=0.7)
       
       ax.set_xlabel('Time')
       ax.set_ylabel('Performance')
       ax.set_title('Model Performance Monitoring')
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
   
   def generate_monitoring_report(self):
       """Generate comprehensive monitoring report"""
       if not self.performance_history:
           return "No performance data available"
       
       current_performance = self.performance_history[-1]['performance']
       total_drift_alerts = len(self.drift_alerts)
       avg_performance = np.mean([entry['performance'] for entry in self.performance_history])
       performance_trend = self._calculate_trend()
       
       report = f"""
📊 Model Performance Monitoring Report
=====================================

Baseline Performance: {self.baseline_performance:.4f}
Current Performance:  {current_performance:.4f}
Performance Change:   {current_performance - self.baseline_performance:+.4f}

Average Performance:  {avg_performance:.4f}
Performance Trend:    {performance_trend}

Total Data Points:    {len(self.performance_history)}
Drift Alerts:         {total_drift_alerts}
Alert Rate:          {total_drift_alerts/len(self.performance_history):.2%}

Recent Drift Alerts:
{self._format_recent_alerts()}

Recommendations:
{self._generate_monitoring_recommendations()}
       """
       
       return report
   
   def _calculate_trend(self):
       """Calculate performance trend"""
       if len(self.performance_history) < 3:
           return "Insufficient data"
       
       recent_performances = [entry['performance'] for entry in self.performance_history[-5:]]
       trend_slope = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
       
       if trend_slope > 0.01:
           return "Improving"
       elif trend_slope < -0.01:
           return "Declining"
       else:
           return "Stable"
   
   def _format_recent_alerts(self):
       """Format recent drift alerts for report"""
       if not self.drift_alerts:
           return "None"
       
       recent_alerts = self.drift_alerts[-3:]  # Last 3 alerts
       formatted = []
       
       for alert in recent_alerts:
           formatted.append(f"  - Timestamp {alert['timestamp']}: "
                          f"Drop of {alert['performance_drop']:.4f} "
                          f"({alert['severity']} severity)")
       
       return "\n".join(formatted)
   
   def _generate_monitoring_recommendations(self):
       """Generate recommendations based on monitoring results"""
       recommendations = []
       
       if len(self.drift_alerts) > 0:
           recent_alert_rate = len([a for a in self.drift_alerts 
                                  if a['timestamp'] >= len(self.performance_history) - 10]) / min(10, len(self.performance_history))
           
           if recent_alert_rate > 0.3:
               recommendations.append("High alert rate detected - consider model retraining")
           
           if any(alert['severity'] in ['Critical', 'High'] for alert in self.drift_alerts[-3:]):
               recommendations.append("Severe drift detected - immediate model update recommended")
       
       current_performance = self.performance_history[-1]['performance']
       if current_performance < self.baseline_performance - 0.1:
           recommendations.append("Significant performance degradation - investigate data quality")
       
       trend = self._calculate_trend()
       if trend == "Declining":
           recommendations.append("Declining performance trend - monitor closely")
       
       if not recommendations:
           recommendations.append("Performance within acceptable range - continue monitoring")
       
       return "\n".join(f"  - {rec}" for rec in recommendations)

# ==========================================
# COMPREHENSIVE EXAMPLE USAGE
# ==========================================

def run_comprehensive_model_evaluation_example():
   """
   Comprehensive example demonstrating all evaluation frameworks
   """
   print("🚀 Neural Learning Web - Week 22: Model Evaluation and Selection")
   print("="*80)
   
   # ================================================================
   # DATA PREPARATION
   # ================================================================
   
   print("\n📊 DATASET PREPARATION")
   print("="*50)
   
   # Create synthetic classification dataset
   X, y = make_classification(
       n_samples=1000, n_features=20, n_informative=15, 
       n_redundant=3, n_clusters_per_class=1, random_state=42
   )
   
   # Use real dataset for demonstration
   from sklearn.datasets import load_breast_cancer
   X, y = load_breast_cancer(return_X_y=True)
   
   print(f"Dataset: Breast Cancer Classification")
   print(f"Samples: {X.shape[0]}")
   print(f"Features: {X.shape[1]}")
   print(f"Classes: {len(np.unique(y))}")
   
   # Scale features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   # ================================================================
   # SINGLE MODEL EVALUATION
   # ================================================================
   
   print("\n🔍 SINGLE MODEL EVALUATION")
   print("="*50)
   
   evaluator = ModelEvaluationFramework('classification')
   
   # Evaluate Random Forest
   rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_results = evaluator.evaluate_single_model(rf_model, X_scaled, y, cv_folds=5)
   
   # ================================================================
   # MULTIPLE MODEL COMPARISON
   # ================================================================
   
   print("\n🏆 MULTIPLE MODEL COMPARISON")
   print("="*50)
   
   models_to_compare = {
       'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
       'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
       'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
       'SVM (RBF)': SVC(random_state=42, probability=True),
       'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
   }
   
   comparison_results = evaluator.compare_multiple_models(
       models_to_compare, X_scaled, y, cv_folds=5
   )
   
   # ================================================================
   # STATISTICAL SIGNIFICANCE TESTING
   # ================================================================
   
   print("\n🔬 STATISTICAL SIGNIFICANCE TESTING")
   print("="*50)
   
   tester = StatisticalSignificanceTester(alpha=0.05)
   
   # Compare top two models
   model_names = list(comparison_results.keys())
   primary_metric = 'accuracy'
   
   # Get top two models by performance
   sorted_models = sorted(
       model_names, 
       key=lambda k: comparison_results[k]['metrics_summary'][primary_metric]['test_mean'],
       reverse=True
   )
   
   best_model = sorted_models[0]
   second_model = sorted_models[1]
   
   best_scores = comparison_results[best_model]['metrics_summary'][primary_metric]['test_scores']
   second_scores = comparison_results[second_model]['metrics_summary'][primary_metric]['test_scores']
   
   # Perform statistical tests
   ttest_result = tester.paired_ttest(best_scores, second_scores, best_model, second_model)
   wilcoxon_result = tester.wilcoxon_test(best_scores, second_scores, best_model, second_model)
   
   # Friedman test for all models
   all_scores = {name: comparison_results[name]['metrics_summary'][primary_metric]['test_scores'] 
                 for name in model_names}
   friedman_result = tester.friedman_test(all_scores)
   
   # ================================================================
   # ADVANCED CROSS-VALIDATION
   # ================================================================
   
   print("\n🔄 ADVANCED CROSS-VALIDATION STRATEGIES")
   print("="*50)
   
   cv_framework = AdvancedCrossValidation()
   
   # Compare different CV strategies
   best_model_instance = models_to_compare[best_model]
   cv_comparison = cv_framework.compare_cv_strategies(X_scaled, y, best_model_instance)
   
   # Nested cross-validation for hyperparameter tuning
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [3, 5, 10, None],
       'min_samples_split': [2, 5, 10]
   }
   
   if best_model == 'Random Forest':
       nested_cv_results = cv_framework.nested_cross_validation(
           X_scaled, y, RandomForestClassifier(random_state=42), param_grid, 
           outer_cv=5, inner_cv=3
       )
   
   # ================================================================
   # AUTOMATED MODEL SELECTION
   # ================================================================
   
   print("\n🤖 AUTOMATED MODEL SELECTION PIPELINE")
   print("="*50)
   
   auto_selector = AutomatedModelSelection('classification')
   automated_results = auto_selector.run_automated_selection(
       X_scaled, y, cv_folds=5, top_k=3
   )
   
   # ================================================================
   # PERFORMANCE MONITORING SIMULATION
   # ================================================================
   
   print("\n📈 PERFORMANCE MONITORING SIMULATION")
   print("="*50)
   
   # Get baseline performance from best model
   baseline_performance = comparison_results[best_model]['metrics_summary'][primary_metric]['test_mean']
   
   monitor = ModelPerformanceMonitor(baseline_performance, threshold=0.03)
   
   # Simulate performance over time with gradual drift
   np.random.seed(42)
   for i in range(20):
       # Simulate gradual performance degradation
       drift_factor = 0.002 * i  # Gradual drift
       noise = np.random.normal(0, 0.01)  # Random noise
       
       simulated_performance = baseline_performance - drift_factor + noise
       monitor.update_performance(simulated_performance, timestamp=i)
   
   # Plot monitoring results
   monitor.plot_performance_history()
   
   # Generate monitoring report
   monitoring_report = monitor.generate_monitoring_report()
   print(monitoring_report)
   
   # ================================================================
   # FINAL SUMMARY AND RECOMMENDATIONS
   # ================================================================
   
   print("\n🎯 FINAL SUMMARY AND RECOMMENDATIONS")
   print("="*50)
   
   print(f"Best Model: {automated_results['final_recommendation']['recommended_model']}")
   print(f"Performance: {automated_results['final_recommendation']['performance_score']:.4f}")
   print(f"Confidence: {automated_results['final_recommendation']['confidence_level']}")
   print(f"Reasoning: {automated_results['final_recommendation']['reasoning']}")
   
   print(f"\nKey Insights:")
   print(f"- Statistical significance testing provides robust model comparison")
   print(f"- Nested CV gives unbiased performance estimates")
   print(f"- Performance monitoring is crucial for production deployment")
   print(f"- Automated pipelines streamline the model selection process")
   
   return {
       'single_model_results': rf_results,
       'comparison_results': comparison_results,
       'significance_results': {
           'ttest': ttest_result,
           'wilcoxon': wilcoxon_result,
           'friedman': friedman_result
       },
       'cv_comparison': cv_comparison,
       'nested_cv_results': nested_cv_results if 'nested_cv_results' in locals() else None,
       'automated_results': automated_results,
       'monitoring_report': monitoring_report
   }

# ==========================================
# EXERCISE CHALLENGES
# ==========================================

def exercise_1_custom_evaluation_metric():
   """
   Exercise 1: Implement custom evaluation metrics for specific business contexts
   """
   print("\n🎯 Exercise 1: Custom Evaluation Metrics")
   print("="*50)
   
   def business_value_score(y_true, y_pred, cost_fp=1.0, cost_fn=5.0, revenue_tp=10.0):
       """
       Custom metric that incorporates business costs and benefits
       """
       from sklearn.metrics import confusion_matrix
       
       tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
       
       # Calculate business value
       total_cost = fp * cost_fp + fn * cost_fn
       total_revenue = tp * revenue_tp
       net_value = total_revenue - total_cost
       
       # Normalize by total samples
       return net_value / len(y_true)
   
   def precision_at_k(y_true, y_prob, k=0.1):
       """
       Precision at top k% of predictions
       """
       n_samples = len(y_true)
       k_samples = int(n_samples * k)
       
       # Get indices of top k predictions
       top_k_indices = np.argsort(y_prob[:, 1])[-k_samples:]
       
       # Calculate precision for top k
       top_k_true = y_true[top_k_indices]
       return np.mean(top_k_true)
   
   # Demonstrate usage
   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)
   
   y_pred = model.predict(X_test)
   y_prob = model.predict_proba(X_test)
   
   business_score = business_value_score(y_test, y_pred)
   precision_at_10 = precision_at_k(y_test, y_prob, k=0.1)
   
   print(f"Business Value Score: {business_score:.2f}")
   print(f"Precision at 10%: {precision_at_10:.4f}")
   print(f"Standard Accuracy: {accuracy_score(y_test, y_pred):.4f}")
   
   return {'business_score': business_score, 'precision_at_k': precision_at_10}

def exercise_2_time_series_evaluation():
   """
   Exercise 2: Implement time series specific evaluation techniques
   """
   print("\n🎯 Exercise 2: Time Series Cross-Validation")
   print("="*50)
   
   # Create time series data
   np.random.seed(42)
   n_samples = 1000
   time_index = np.arange(n_samples)
   
   # Simulate time series with trend and seasonality
   trend = 0.01 * time_index
   seasonality = 2 * np.sin(2 * np.pi * time_index / 50)
   noise = np.random.normal(0, 0.5, n_samples)
   
   y_ts = trend + seasonality + noise
   X_ts = np.column_stack([time_index, np.sin(time_index), np.cos(time_index)])
   
   # Time series cross-validation
   tscv = TimeSeriesSplit(n_splits=5)
   
   model = Ridge(alpha=1.0)
   
   print(f"Time Series Cross-Validation:")
   scores = []
   
   for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
       X_train_ts, X_test_ts = X_ts[train_idx], X_ts[test_idx]
       y_train_ts, y_test_ts = y_ts[train_idx], y_ts[test_idx]
       
       model.fit(X_train_ts, y_train_ts)
       score = model.score(X_test_ts, y_test_ts)
       scores.append(score)
       
       print(f"  Fold {fold + 1}: Train size {len(train_idx)}, Test size {len(test_idx)}, R² = {score:.4f}")
   
   print(f"Mean R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
   
   return {'ts_scores': scores, 'ts_mean': np.mean(scores)}

def exercise_3_model_interpretability_evaluation():
   """
   Exercise 3: Evaluate model interpretability alongside performance
   """
   print("\n🎯 Exercise 3: Model Interpretability Evaluation")
   print("="*50)
   
   from sklearn.inspection import permutation_importance
   from sklearn.tree import DecisionTreeClassifier
   
   # Load dataset
   X, y = load_breast_cancer(return_X_y=True)
   feature_names = load_breast_cancer().feature_names
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
   # Compare interpretable vs complex models
   models = {
       'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
       'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
       'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
   }
   
   interpretability_results = {}
   
   for name, model in models.items():
       model.fit(X_train, y_train)
       
       # Performance metrics
       train_acc = model.score(X_train, y_train)
       test_acc = model.score(X_test, y_test)
       
       # Feature importance
       if hasattr(model, 'feature_importances_'):
           feature_importance = model.feature_importances_
       else:
           # Use permutation importance for models without built-in importance
           perm_importance = permutation_importance(model, X_test, y_test, random_state=42)
           feature_importance = perm_importance.importances_mean
       
       # Interpretability score (based on model complexity)
       if name == 'Decision Tree':
           interpretability_score = 0.9  # High interpretability
       elif name == 'Logistic Regression':
           interpretability_score = 0.8  # Good interpretability
       else:
           interpretability_score = 0.3  # Low interpretability
       
       interpretability_results[name] = {
           'train_accuracy': train_acc,
           'test_accuracy': test_acc,
           'interpretability_score': interpretability_score,
           'feature_importance': feature_importance,
           'top_features': np.argsort(feature_importance)[-5:][::-1]
       }
       
       print(f"{name}:")
       print(f"  Test Accuracy: {test_acc:.4f}")
       print(f"  Interpretability Score: {interpretability_score:.1f}")
       print(f"  Top 3 Features: {[feature_names[i] for i in interpretability_results[name]['top_features'][:3]]}")
       print()
   
   return interpretability_results

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
   print("🧠 Neural Learning Web - Phase 2, Week 22")
   print("Model Evaluation and Selection - Comprehensive Implementation")
   print("="*80)
   
   # Run comprehensive example
   comprehensive_results = run_comprehensive_model_evaluation_example()
   
   print("\n" + "="*80)
   print("📚 EXERCISES")
   print("="*80)
   
   # Run exercises
   exercise_1_results = exercise_1_custom_evaluation_metric()
   exercise_2_results = exercise_2_time_series_evaluation()
   exercise_3_results = exercise_3_model_interpretability_evaluation()
   
   print("\n🎉 Week 22 Complete!")
   print("You've mastered comprehensive model evaluation and selection techniques!")
   print("Ready to move on to advanced ensemble methods and meta-learning approaches.")