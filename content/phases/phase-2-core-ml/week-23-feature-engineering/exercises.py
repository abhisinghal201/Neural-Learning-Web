"""
Neural Learning Web - Phase 2: Week 23 - Feature Engineering
===========================================================

Week 23: Feature Engineering - The Art and Science of Creating Predictive Features
Master the crucial skill that often separates good ML practitioners from great ones.
Feature engineering is where domain knowledge meets algorithmic creativity.

Building on your solid foundation in model evaluation and selection, this week focuses
on the systematic creation, transformation, and selection of features that maximize
model performance while maintaining interpretability and robustness.

Learning Objectives:
- Master systematic approaches to feature creation and transformation
- Implement advanced feature selection techniques with statistical validation
- Build automated feature engineering pipelines for different data types
- Design domain-specific feature extraction strategies
- Create feature interaction and polynomial feature systems
- Develop time series and sequential feature engineering frameworks
- Build interpretable feature importance and selection systems

Author: Neural Explorer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_boston, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
                                 PowerTransformer, PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                 OrdinalEncoder, TargetEncoder)
from sklearn.feature_selection import (SelectKBest, SelectPercentile, RFE, RFECV,
                                     f_classif, f_regression, mutual_info_classif, 
                                     mutual_info_regression, chi2, VarianceThreshold)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from scipy import stats
from scipy.special import boxcox1p
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# SYSTEMATIC FEATURE CREATION FRAMEWORK
# ==========================================

class FeatureCreationFramework:
    """
    Comprehensive framework for systematic feature creation and transformation
    """
    
    def __init__(self):
        self.feature_history = []
        self.transformation_log = {}
        self.performance_tracker = {}
        
    def create_mathematical_features(self, X, feature_names=None):
        """
        Create mathematical transformations of existing features
        """
        print("🔢 Creating Mathematical Features")
        print("="*50)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        new_features = pd.DataFrame()
        
        print(f"Original features: {X.shape[1]}")
        
        # 1. Polynomial features (degree 2)
        print("   Creating polynomial features...")
        for i, col1 in enumerate(feature_names):
            for j, col2 in enumerate(feature_names[i:], i):
                if i == j:
                    # Squared terms
                    new_features[f'{col1}_squared'] = X_df[col1] ** 2
                else:
                    # Interaction terms
                    new_features[f'{col1}_x_{col2}'] = X_df[col1] * X_df[col2]
        
        # 2. Mathematical transformations
        print("   Creating mathematical transformations...")
        for col in feature_names:
            # Handle negative values by shifting
            min_val = X_df[col].min()
            if min_val <= 0:
                shifted_col = X_df[col] - min_val + 1e-8
            else:
                shifted_col = X_df[col]
            
            # Logarithmic
            new_features[f'{col}_log'] = np.log(shifted_col)
            
            # Square root
            new_features[f'{col}_sqrt'] = np.sqrt(np.abs(X_df[col]))
            
            # Reciprocal (avoid division by zero)
            new_features[f'{col}_reciprocal'] = 1 / (X_df[col] + 1e-8)
            
            # Exponential (clipped to avoid overflow)
            new_features[f'{col}_exp'] = np.exp(np.clip(X_df[col], -10, 2))
        
        # 3. Statistical aggregations across features
        print("   Creating statistical aggregations...")
        new_features['row_sum'] = X_df.sum(axis=1)
        new_features['row_mean'] = X_df.mean(axis=1)
        new_features['row_std'] = X_df.std(axis=1)
        new_features['row_min'] = X_df.min(axis=1)
        new_features['row_max'] = X_df.max(axis=1)
        new_features['row_range'] = new_features['row_max'] - new_features['row_min']
        
        # 4. Ratios between features
        print("   Creating ratio features...")
        for i, col1 in enumerate(feature_names):
            for col2 in feature_names[i+1:]:
                # Avoid division by zero
                denominator = X_df[col2] + 1e-8
                new_features[f'{col1}_div_{col2}'] = X_df[col1] / denominator
        
        print(f"Created {new_features.shape[1]} new mathematical features")
        
        # Combine original and new features
        X_combined = pd.concat([X_df, new_features], axis=1)
        
        self.feature_history.append({
            'step': 'mathematical_features',
            'original_features': X.shape[1],
            'new_features': new_features.shape[1],
            'total_features': X_combined.shape[1]
        })
        
        return X_combined.values, list(X_combined.columns)
    
    def create_binning_features(self, X, feature_names=None, n_bins=5):
        """
        Create binned/discretized versions of continuous features
        """
        print(f"📊 Creating Binning Features ({n_bins} bins)")
        print("="*50)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        new_features = pd.DataFrame()
        
        for col in feature_names:
            if X_df[col].nunique() > n_bins:  # Only bin if there are more unique values than bins
                # Equal-width binning
                binned_equal = pd.cut(X_df[col], bins=n_bins, labels=False)
                new_features[f'{col}_binned_equal'] = binned_equal
                
                # Equal-frequency binning (quantiles)
                binned_quantile = pd.qcut(X_df[col], q=n_bins, labels=False, duplicates='drop')
                new_features[f'{col}_binned_quantile'] = binned_quantile
        
        print(f"Created {new_features.shape[1]} binning features")
        
        # Combine with original features
        X_combined = pd.concat([X_df, new_features], axis=1)
        
        return X_combined.values, list(X_combined.columns)
    
    def create_clustering_features(self, X, feature_names=None, n_clusters_list=[3, 5, 10]):
        """
        Create cluster-based features using different clustering algorithms
        """
        print(f"🎯 Creating Clustering Features")
        print("="*50)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        new_features = pd.DataFrame()
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for n_clusters in n_clusters_list:
            print(f"   Creating features with {n_clusters} clusters...")
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            new_features[f'kmeans_{n_clusters}_cluster'] = clusters
            
            # Distance to each cluster center
            distances = kmeans.transform(X_scaled)
            for i in range(n_clusters):
                new_features[f'kmeans_{n_clusters}_dist_{i}'] = distances[:, i]
            
            # Distance to nearest cluster center
            new_features[f'kmeans_{n_clusters}_min_dist'] = np.min(distances, axis=1)
        
        print(f"Created {new_features.shape[1]} clustering features")
        
        # Combine with original features
        X_combined = pd.concat([X_df, new_features], axis=1)
        
        return X_combined.values, list(X_combined.columns)

# ==========================================
# ADVANCED FEATURE SELECTION FRAMEWORK
# ==========================================

class AdvancedFeatureSelection:
    """
    Advanced feature selection with multiple strategies and statistical validation
    """
    
    def __init__(self):
        self.selection_results = {}
        self.feature_importance_history = []
        
    def univariate_selection_comprehensive(self, X, y, problem_type='classification'):
        """
        Comprehensive univariate feature selection with multiple scoring functions
        """
        print("🔍 Comprehensive Univariate Feature Selection")
        print("="*50)
        
        results = {}
        
        if problem_type == 'classification':
            scoring_functions = {
                'f_classif': f_classif,
                'mutual_info': mutual_info_classif,
                'chi2': chi2 if np.all(X >= 0) else None  # chi2 requires non-negative features
            }
        else:
            scoring_functions = {
                'f_regression': f_regression,
                'mutual_info': mutual_info_regression
            }
        
        for score_name, score_func in scoring_functions.items():
            if score_func is None:
                continue
                
            print(f"   Applying {score_name}...")
            
            if 'mutual_info' in score_name:
                scores = score_func(X, y, random_state=42)
            else:
                scores, p_values = score_func(X, y)
                results[f'{score_name}_pvalues'] = p_values
            
            results[f'{score_name}_scores'] = scores
            
            # Select top features
            top_k = min(20, X.shape[1] // 2)  # Select up to 20 or half of features
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[f'{score_name}_top_features'] = top_indices
            
            print(f"      Top 5 features: {top_indices[:5].tolist()}")
            print(f"      Top 5 scores: {scores[top_indices[:5]]}")
        
        self.selection_results['univariate'] = results
        return results
    
    def multivariate_selection(self, X, y, problem_type='classification'):
        """
        Multivariate feature selection using various algorithms
        """
        print("🎭 Multivariate Feature Selection")
        print("="*50)
        
        results = {}
        
        # Choose base estimator based on problem type
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # 1. Recursive Feature Elimination with Cross-Validation
        print("   Performing RFE with Cross-Validation...")
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy' if problem_type == 'classification' else 'r2',
                      min_features_to_select=5)
        rfecv.fit(X, y)
        
        results['rfecv'] = {
            'selected_features': np.where(rfecv.support_)[0],
            'ranking': rfecv.ranking_,
            'cv_scores': rfecv.cv_results_['mean_test_score'],
            'optimal_features': rfecv.n_features_
        }
        
        print(f"      Optimal number of features: {rfecv.n_features_}")
        print(f"      Selected features: {results['rfecv']['selected_features'][:10].tolist()}")
        
        # 2. Tree-based feature importance
        print("   Computing tree-based feature importance...")
        estimator.fit(X, y)
        feature_importance = estimator.feature_importances_
        
        # Select features above median importance
        importance_threshold = np.median(feature_importance)
        important_features = np.where(feature_importance > importance_threshold)[0]
        
        results['tree_importance'] = {
            'feature_importance': feature_importance,
            'selected_features': important_features,
            'threshold': importance_threshold
        }
        
        print(f"      Selected {len(important_features)} features above median importance")
        
        # 3. L1-based feature selection (Lasso)
        print("   Performing L1-based feature selection...")
        if problem_type == 'classification':
            l1_selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
        else:
            l1_selector = Lasso(alpha=0.1, random_state=42)
        
        l1_selector.fit(X, y)
        l1_selected_features = np.where(np.abs(l1_selector.coef_.ravel()) > 1e-8)[0]
        
        results['l1_selection'] = {
            'coefficients': l1_selector.coef_.ravel(),
            'selected_features': l1_selected_features
        }
        
        print(f"      L1 selected {len(l1_selected_features)} features")
        
        self.selection_results['multivariate'] = results
        return results
    
    def stability_selection(self, X, y, problem_type='classification', n_bootstrap=100):
        """
        Stability selection using bootstrap resampling
        """
        print(f"🎪 Stability Selection ({n_bootstrap} bootstrap samples)")
        print("="*50)
        
        n_features = X.shape[1]
        selection_frequencies = np.zeros(n_features)
        
        # Choose selection method
        if problem_type == 'classification':
            selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
        else:
            selector = Lasso(alpha=0.1, random_state=42)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit selector
            selector.fit(X_boot, y_boot)
            
            # Track selected features
            if hasattr(selector, 'coef_'):
                selected = np.abs(selector.coef_.ravel()) > 1e-8
            else:
                selected = selector.support_
            
            selection_frequencies[selected] += 1
        
        # Convert to probabilities
        selection_probabilities = selection_frequencies / n_bootstrap
        
        # Select stable features (selected in at least 60% of bootstrap samples)
        stability_threshold = 0.6
        stable_features = np.where(selection_probabilities >= stability_threshold)[0]
        
        results = {
            'selection_probabilities': selection_probabilities,
            'stable_features': stable_features,
            'threshold': stability_threshold
        }
        
        print(f"   Selected {len(stable_features)} stable features")
        print(f"   Stability threshold: {stability_threshold}")
        print(f"   Top 5 most stable features: {np.argsort(selection_probabilities)[::-1][:5].tolist()}")
        
        self.selection_results['stability'] = results
        return results
    
    def compare_selection_methods(self, X, y, problem_type='classification'):
        """
        Compare all feature selection methods and analyze overlap
        """
        print("🤝 Comparing Feature Selection Methods")
        print("="*50)
        
        # Run all selection methods
        univariate_results = self.univariate_selection_comprehensive(X, y, problem_type)
        multivariate_results = self.multivariate_selection(X, y, problem_type)
        stability_results = self.stability_selection(X, y, problem_type)
        
        # Extract selected features from each method
        method_features = {}
        
        # Univariate methods
        for method_name in ['f_classif', 'mutual_info', 'f_regression']:
            if f'{method_name}_top_features' in univariate_results:
                method_features[method_name] = set(univariate_results[f'{method_name}_top_features'][:10])
        
        # Multivariate methods
        method_features['rfecv'] = set(multivariate_results['rfecv']['selected_features'])
        method_features['tree_importance'] = set(multivariate_results['tree_importance']['selected_features'])
        method_features['l1_selection'] = set(multivariate_results['l1_selection']['selected_features'])
        method_features['stability'] = set(stability_results['stable_features'])
        
        # Analyze overlaps
        print("\n🔗 Feature Selection Overlap Analysis:")
        
        method_names = list(method_features.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                overlap = method_features[method1].intersection(method_features[method2])
                union = method_features[method1].union(method_features[method2])
                jaccard = len(overlap) / len(union) if len(union) > 0 else 0
                
                print(f"   {method1} ∩ {method2}: {len(overlap)} features (Jaccard: {jaccard:.3f})")
        
        # Find consensus features (selected by majority of methods)
        all_selected_features = []
        for features in method_features.values():
            all_selected_features.extend(list(features))
        
        feature_counts = pd.Series(all_selected_features).value_counts()
        consensus_threshold = len(method_features) // 2
        consensus_features = feature_counts[feature_counts >= consensus_threshold].index.tolist()
        
        print(f"\n🎯 Consensus features (selected by ≥{consensus_threshold} methods): {len(consensus_features)}")
        print(f"   Features: {consensus_features}")
        
        # Visualize selection comparison
        self._visualize_selection_comparison(method_features, consensus_features)
        
        return {
            'method_features': method_features,
            'consensus_features': consensus_features,
            'univariate_results': univariate_results,
            'multivariate_results': multivariate_results,
            'stability_results': stability_results
        }
    
    def _visualize_selection_comparison(self, method_features, consensus_features):
        """
        Visualize feature selection comparison results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Selection frequency heatmap
        ax = axes[0, 0]
        methods = list(method_features.keys())
        all_features = set()
        for features in method_features.values():
            all_features.update(features)
        all_features = sorted(list(all_features))
        
        # Create binary matrix
        selection_matrix = np.zeros((len(methods), len(all_features)))
        for i, method in enumerate(methods):
            for j, feature in enumerate(all_features):
                if feature in method_features[method]:
                    selection_matrix[i, j] = 1
        
        # Show only first 20 features for readability
        max_features = min(20, len(all_features))
        im = ax.imshow(selection_matrix[:, :max_features], cmap='RdYlBu', aspect='auto')
        ax.set_xticks(range(max_features))
        ax.set_xticklabels([f'F{f}' for f in all_features[:max_features]], rotation=45)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title('Feature Selection by Method')
        plt.colorbar(im, ax=ax)
        
        # 2. Number of features selected by each method
        ax = axes[0, 1]
        method_counts = [len(features) for features in method_features.values()]
        bars = ax.bar(methods, method_counts, alpha=0.7, color='skyblue')
        ax.set_title('Number of Features Selected by Method')
        ax.set_xlabel('Selection Method')
        ax.set_ylabel('Number of Features')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, method_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        # 3. Consensus features visualization
        ax = axes[1, 0]
        if consensus_features:
            ax.bar(range(len(consensus_features)), [1]*len(consensus_features), 
                  alpha=0.7, color='green')
            ax.set_title(f'Consensus Features ({len(consensus_features)} features)')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Selected')
            ax.set_xticks(range(len(consensus_features)))
            ax.set_xticklabels([f'F{f}' for f in consensus_features], rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Consensus Features', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Consensus Features')
        
        # 4. Method overlap network (simplified)
        ax = axes[1, 1]
        # Create overlap matrix
        n_methods = len(methods)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    overlap = len(method_features[methods[i]].intersection(method_features[methods[j]]))
                    union = len(method_features[methods[i]].union(method_features[methods[j]]))
                    jaccard = overlap / union if union > 0 else 0
                    overlap_matrix[i, j] = jaccard
        
        im = ax.imshow(overlap_matrix, cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels(methods)
        ax.set_title('Method Overlap (Jaccard Similarity)')
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# DOMAIN-SPECIFIC FEATURE ENGINEERING
# ==========================================

class DomainSpecificFeatures:
    """
    Domain-specific feature engineering for common data types
    """
    
    def __init__(self):
        self.feature_extractors = {}
    
    def text_features(self, texts, max_features=1000):
        """
        Extract features from text data
        """
        print("📝 Extracting Text Features")
        print("="*50)
        
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from textstat import flesch_reading_ease, flesch_kincaid_grade
        
        # Basic text statistics
        text_stats = pd.DataFrame()
        text_stats['char_count'] = [len(text) for text in texts]
        text_stats['word_count'] = [len(text.split()) for text in texts]
        text_stats['sentence_count'] = [text.count('.') + text.count('!') + text.count('?') for text in texts]
        text_stats['avg_word_length'] = [np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in texts]
        
        # Readability scores
        try:
            text_stats['flesch_ease'] = [flesch_reading_ease(text) for text in texts]
            text_stats['flesch_grade'] = [flesch_kincaid_grade(text) for text in texts]
        except:
            print("   Textstat not available for readability scores")
        
        # TF-IDF features
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
        tfidf_features = tfidf.fit_transform(texts).toarray()
        
        print(f"   Created {text_stats.shape[1]} statistical features")
        print(f"   Created {tfidf_features.shape[1]} TF-IDF features")
        
        # Combine features
        all_features = np.hstack([text_stats.values, tfidf_features])
        feature_names = list(text_stats.columns) + [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        
        return all_features, feature_names
    
    def time_series_features(self, ts_data, window_sizes=[5, 10, 20]):
        """
        Extract time series features
        """
        print("⏰ Extracting Time Series Features")
        print("="*50)
        
        if isinstance(ts_data, list):
            ts_data = np.array(ts_data)
        
        features = pd.DataFrame()
        
        # Basic statistics
        features['mean'] = [np.mean(ts_data)]
        features['std'] = [np.std(ts_data)]
        features['min'] = [np.min(ts_data)]
        features['max'] = [np.max(ts_data)]
        features['range'] = features['max'] - features['min']
        
        # Trend features
        x = np.arange(len(ts_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data)
        features['trend_slope'] = [slope]
        features['trend_r2'] = [r_value**2]
        
        # Moving window features
        for window in window_sizes:
            if window < len(ts_data):
                rolling_mean = pd.Series(ts_data).rolling(window=window).mean()
                rolling_std = pd.Series(ts_data).rolling(window=window).std()
                
                features[f'ma_{window}_final'] = [rolling_mean.iloc[-1]]
                features[f'std_{window}_final'] = [rolling_std.iloc[-1]]
                features[f'ma_{window}_trend'] = [np.polyfit(range(len(rolling_mean.dropna())), 
                                                           rolling_mean.dropna(), 1)[0]]
        
        # Seasonal features (basic)
        if len(ts_data) >= 12:  # Assume monthly data
            seasonal_mean = np.mean([ts_data[i::12] for i in range(12)], axis=1)
            features['seasonal_variation'] = [np.std(seasonal_mean)]
        
        print(f"   Created {features.shape[1]} time series features")
        
        return features.values[0], list(features.columns)
    
    def image_features(self, images):
        """
        Extract basic features from image data (simplified)
        """
        print("🖼️ Extracting Image Features")
        print("="*50)
        
        features = []
        feature_names = []
        
        for i, img in enumerate(images):
            if len(img.shape) == 3:  # Color image
                # Color channel statistics
                for c, channel in enumerate(['red', 'green', 'blue']):
                    features.extend([
                        np.mean(img[:, :, c]),
                        np.std(img[:, :, c]),
                        np.min(img[:, :, c]),
                        np.max(img[:, :, c])
                    ])
                    feature_names.extend([f'{channel}_mean', f'{channel}_std', 
                                        f'{channel}_min', f'{channel}_max'])
            else:  # Grayscale
                features.extend([
                    np.mean(img),
                    np.std(img),
                    np.min(img),
                    np.max(img)
                ])
                feature_names.extend(['gray_mean', 'gray_std', 'gray_min', 'gray_max'])
            
            # Basic texture features (simplified)
            if len(img.shape) == 2:
                # Gradient magnitude
                gy, gx = np.gradient(img)
                gradient_magnitude = np.sqrt(gx**2 + gy**2)
                features.extend([
                    np.mean(gradient_magnitude),
                    np.std(gradient_magnitude)
                ])
                feature_names.extend(['gradient_mean', 'gradient_std'])
            
            break  # Process only first image for demonstration
        
        print(f"   Created {len(features)} image features")
        
        return np.array(features), feature_names

# ==========================================
# AUTOMATED FEATURE ENGINEERING PIPELINE
# ==========================================

class AutomatedFeatureEngineering:
    """
    Automated feature engineering pipeline with performance tracking
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.pipeline_steps = []
        self.performance_history = []
        self.best_features = None
        self.best_score = -np.inf
        
    def evaluate_features(self, X, y, feature_names=None):
        """
        Evaluate feature set performance using cross-validation
        """
        if self.problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'r2'
        
        # Handle missing values
        X_clean = np.nan_to_num(X, nan=np.nanmean(X))
        
        # Cross-validation
        cv_scores = cross_val_score(estimator, X_clean, y, cv=5, scoring=scoring)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Track performance
        self.performance_history.append({
            'n_features': X.shape[1],
            'mean_score': mean_score,
            'std_score': std_score,
            'feature_names': feature_names[:10] if feature_names else None  # Store first 10 for brevity
        })
        
        # Update best features
        if mean_score > self.best_score:
            self.best_score = mean_score
            self.best_features = X.copy()
        
        return mean_score, std_score
    
    def automated_pipeline(self, X, y, feature_names=None):
        """
        Run automated feature engineering pipeline
        """
        print("🤖 Automated Feature Engineering Pipeline")
        print("="*70)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        current_X = X.copy()
        current_names = feature_names.copy()
        
        # Step 1: Baseline evaluation
        print("\n1️⃣ Baseline Evaluation")
        baseline_score, baseline_std = self.evaluate_features(current_X, y, current_names)
        print(f"   Baseline performance: {baseline_score:.4f} ± {baseline_std:.4f}")
        
        # Step 2: Feature creation
        print("\n2️⃣ Feature Creation Phase")
        
        # Mathematical features
        feature_creator = FeatureCreationFramework()
        X_math, names_math = feature_creator.create_mathematical_features(current_X, current_names)
        math_score, math_std = self.evaluate_features(X_math, y, names_math)
        print(f"   With mathematical features: {math_score:.4f} ± {math_std:.4f}")
        
        if math_score > baseline_score + 0.01:  # Significant improvement
            current_X = X_math
            current_names = names_math
            print("   ✅ Mathematical features improved performance")
        else:
            print("   ❌ Mathematical features didn't improve performance")
        
        # Binning features
        X_binned, names_binned = feature_creator.create_binning_features(current_X, current_names)
        binned_score, binned_std = self.evaluate_features(X_binned, y, names_binned)
        print(f"   With binning features: {binned_score:.4f} ± {binned_std:.4f}")
        
        if binned_score > max(baseline_score, math_score) + 0.01:
            current_X = X_binned
            current_names = names_binned
            print("   ✅ Binning features improved performance")
        else:
            print("   ❌ Binning features didn't improve performance")
        
        # Clustering features
        X_cluster, names_cluster = feature_creator.create_clustering_features(current_X, current_names)
        cluster_score, cluster_std = self.evaluate_features(X_cluster, y, names_cluster)
        print(f"   With clustering features: {cluster_score:.4f} ± {cluster_std:.4f}")
        
        if cluster_score > max(baseline_score, math_score, binned_score) + 0.01:
            current_X = X_cluster
            current_names = names_cluster
            print("   ✅ Clustering features improved performance")
        else:
            print("   ❌ Clustering features didn't improve performance")
        
        # Step 3: Feature selection
        print("\n3️⃣ Feature Selection Phase")
        
        if current_X.shape[1] > 50:  # Only if we have many features
            feature_selector = AdvancedFeatureSelection()
            selection_results = feature_selector.compare_selection_methods(
                current_X, y, self.problem_type
            )
            
            # Use consensus features
            consensus_features = selection_results['consensus_features']
            if len(consensus_features) > 10:  # Ensure we have enough features
                X_selected = current_X[:, consensus_features]
                names_selected = [current_names[i] for i in consensus_features]
                
                selected_score, selected_std = self.evaluate_features(X_selected, y, names_selected)
                print(f"   With feature selection: {selected_score:.4f} ± {selected_std:.4f}")
                
                if selected_score > cluster_score - 0.02:  # Allow small decrease for simplicity
                    current_X = X_selected
                    current_names = names_selected
                    print("   ✅ Feature selection maintained performance with fewer features")
                else:
                    print("   ❌ Feature selection reduced performance too much")
        
        # Step 4: Final evaluation and comparison
        print("\n4️⃣ Final Results")
        final_score, final_std = self.evaluate_features(current_X, y, current_names)
        improvement = final_score - baseline_score
        
        print(f"   Baseline:  {baseline_score:.4f} ± {baseline_std:.4f}")
        print(f"   Final:     {final_score:.4f} ± {final_std:.4f}")
        print(f"   Improvement: {improvement:+.4f}")
        print(f"   Features: {X.shape[1]} → {current_X.shape[1]}")
        
        # Visualize pipeline results
        self._visualize_pipeline_results()
        
        return current_X, current_names, {
            'baseline_score': baseline_score,
            'final_score': final_score,
            'improvement': improvement,
            'original_features': X.shape[1],
            'final_features': current_X.shape[1]
        }
    
    def _visualize_pipeline_results(self):
        """
        Visualize the automated pipeline results
        """
        if not self.performance_history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance over pipeline steps
        ax = axes[0]
        steps = range(len(self.performance_history))
        scores = [h['mean_score'] for h in self.performance_history]
        stds = [h['std_score'] for h in self.performance_history]
        
        ax.errorbar(steps, scores, yerr=stds, marker='o', capsize=5, capthick=2)
        ax.set_xlabel('Pipeline Step')
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Performance Throughout Pipeline')
        ax.grid(True, alpha=0.3)
        
        # Add step labels
        step_labels = ['Baseline', 'Math Features', 'Binning', 'Clustering', 'Selection']
        ax.set_xticks(steps)
        ax.set_xticklabels(step_labels[:len(steps)], rotation=45, ha='right')
        
        # Feature count over pipeline steps
        ax = axes[1]
        feature_counts = [h['n_features'] for h in self.performance_history]
        bars = ax.bar(steps, feature_counts, alpha=0.7, color='skyblue')
        ax.set_xlabel('Pipeline Step')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Count Throughout Pipeline')
        ax.set_xticks(steps)
        ax.set_xticklabels(step_labels[:len(steps)], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# FEATURE INTERACTION AND POLYNOMIAL FEATURES
# ==========================================

class FeatureInteractionFramework:
    """
    Advanced framework for feature interactions and polynomial features
    """
    
    def __init__(self):
        self.interaction_scores = {}
        self.polynomial_results = {}
        
    def detect_feature_interactions(self, X, y, feature_names=None, max_interactions=20):
        """
        Detect important feature interactions using statistical tests
        """
        print("🔗 Detecting Feature Interactions")
        print("="*50)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        n_features = X.shape[1]
        interactions = []
        
        print(f"   Evaluating {n_features*(n_features-1)//2} potential interactions...")
        
        # Test all pairwise interactions
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Create interaction term
                interaction_term = X[:, i] * X[:, j]
                
                # Test if interaction improves prediction
                X_with_interaction = np.column_stack([X, interaction_term])
                
                # Use simple correlation or mutual information
                if len(np.unique(y)) == 2:  # Binary classification
                    score = np.abs(np.corrcoef(interaction_term, y)[0, 1])
                else:  # Regression or multi-class
                    score = np.abs(np.corrcoef(interaction_term, y)[0, 1])
                
                if not np.isnan(score):
                    interactions.append({
                        'feature1': i,
                        'feature2': j,
                        'feature1_name': feature_names[i],
                        'feature2_name': feature_names[j],
                        'score': score
                    })
        
        # Sort by score and select top interactions
        interactions.sort(key=lambda x: x['score'], reverse=True)
        top_interactions = interactions[:max_interactions]
        
        print(f"   Top 5 interactions:")
        for i, interaction in enumerate(top_interactions[:5]):
            print(f"      {i+1}. {interaction['feature1_name']} × {interaction['feature2_name']}: {interaction['score']:.4f}")
        
        self.interaction_scores = top_interactions
        return top_interactions
    
    def create_polynomial_features(self, X, degree=2, interaction_only=False, feature_names=None):
        """
        Create polynomial features with performance evaluation
        """
        print(f"📈 Creating Polynomial Features (degree {degree})")
        print("="*50)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                                include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(feature_names)
        
        print(f"   Original features: {X.shape[1]}")
        print(f"   Polynomial features: {X_poly.shape[1]}")
        print(f"   New features created: {X_poly.shape[1] - X.shape[1]}")
        
        # Analyze feature importance
        if X_poly.shape[1] < 1000:  # Only if manageable size
            feature_correlations = []
            for i in range(X.shape[1], X_poly.shape[1]):  # Only new features
                corr = np.corrcoef(X_poly[:, i], X_poly[:, 0])[0, 1]  # Correlation with first original feature
                if not np.isnan(corr):
                    feature_correlations.append((i, poly_feature_names[i], abs(corr)))
            
            # Sort by correlation strength
            feature_correlations.sort(key=lambda x: x[2], reverse=True)
            
            print(f"   Top 5 polynomial features by correlation:")
            for i, (idx, name, corr) in enumerate(feature_correlations[:5]):
                print(f"      {i+1}. {name}: {corr:.4f}")
        
        self.polynomial_results = {
            'X_poly': X_poly,
            'feature_names': poly_feature_names,
            'degree': degree,
            'interaction_only': interaction_only
        }
        
        return X_poly, poly_feature_names
    
    def evaluate_interaction_impact(self, X, y, top_k_interactions=10, problem_type='classification'):
        """
        Evaluate the impact of adding top interactions to the model
        """
        print(f"⚡ Evaluating Interaction Impact (top {top_k_interactions})")
        print("="*50)
        
        if not self.interaction_scores:
            print("   No interactions detected. Run detect_feature_interactions first.")
            return None
        
        # Baseline performance
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'r2'
        
        baseline_scores = cross_val_score(estimator, X, y, cv=5, scoring=scoring)
        baseline_mean = np.mean(baseline_scores)
        
        # Create dataset with top interactions
        X_with_interactions = X.copy()
        interaction_names = []
        
        for i, interaction in enumerate(self.interaction_scores[:top_k_interactions]):
            interaction_term = X[:, interaction['feature1']] * X[:, interaction['feature2']]
            X_with_interactions = np.column_stack([X_with_interactions, interaction_term])
            interaction_names.append(f"{interaction['feature1_name']}_x_{interaction['feature2_name']}")
        
        # Evaluate with interactions
        interaction_scores = cross_val_score(estimator, X_with_interactions, y, cv=5, scoring=scoring)
        interaction_mean = np.mean(interaction_scores)
        
        improvement = interaction_mean - baseline_mean
        
        print(f"   Baseline performance: {baseline_mean:.4f} ± {np.std(baseline_scores):.4f}")
        print(f"   With interactions:    {interaction_mean:.4f} ± {np.std(interaction_scores):.4f}")
        print(f"   Improvement:          {improvement:+.4f}")
        print(f"   Features added:       {top_k_interactions}")
        
        return {
            'baseline_score': baseline_mean,
            'interaction_score': interaction_mean,
            'improvement': improvement,
            'X_with_interactions': X_with_interactions,
            'interaction_names': interaction_names
        }

# ==========================================
# DIMENSIONALITY REDUCTION FRAMEWORK
# ==========================================

class DimensionalityReductionFramework:
    """
    Comprehensive dimensionality reduction with multiple techniques
    """
    
    def __init__(self):
        self.reduction_results = {}
        
    def apply_linear_reductions(self, X, n_components=10):
        """
        Apply linear dimensionality reduction techniques
        """
        print("📉 Applying Linear Dimensionality Reduction")
        print("="*50)
        
        results = {}
        
        # PCA
        print("   Applying PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        
        results['pca'] = {
            'X_reduced': X_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'components': pca.components_,
            'transformer': pca
        }
        
        print(f"      Explained variance ratio: {explained_variance_ratio:.4f}")
        
        # Truncated SVD (good for sparse data)
        print("   Applying Truncated SVD...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_svd = svd.fit_transform(X)
        svd_explained_variance_ratio = np.sum(svd.explained_variance_ratio_)
        
        results['svd'] = {
            'X_reduced': X_svd,
            'explained_variance_ratio': svd_explained_variance_ratio,
            'components': svd.components_,
            'transformer': svd
        }
        
        print(f"      Explained variance ratio: {svd_explained_variance_ratio:.4f}")
        
        # Factor Analysis
        print("   Applying Factor Analysis...")
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        X_fa = fa.fit_transform(X)
        
        results['fa'] = {
            'X_reduced': X_fa,
            'components': fa.components_,
            'transformer': fa
        }
        
        # ICA
        print("   Applying Independent Component Analysis...")
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        try:
            X_ica = ica.fit_transform(X)
            results['ica'] = {
                'X_reduced': X_ica,
                'components': ica.components_,
                'transformer': ica
            }
        except:
            print("      ICA failed to converge")
        
        self.reduction_results['linear'] = results
        return results
    
    def apply_nonlinear_reductions(self, X, n_components=2):
        """
        Apply nonlinear dimensionality reduction techniques
        """
        print("🌀 Applying Nonlinear Dimensionality Reduction")
        print("="*50)
        
        # Limit data size for computational efficiency
        if X.shape[0] > 1000:
            print("   Sampling 1000 points for nonlinear methods...")
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        results = {}
        
        # t-SNE
        print("   Applying t-SNE...")
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X_sample.shape[0]-1))
        X_tsne = tsne.fit_transform(X_sample)
        
        results['tsne'] = {
            'X_reduced': X_tsne,
            'transformer': tsne
        }
        
        # Isomap
        print("   Applying Isomap...")
        try:
            isomap = Isomap(n_components=n_components, n_neighbors=min(10, X_sample.shape[0]-1))
            X_isomap = isomap.fit_transform(X_sample)
            
            results['isomap'] = {
                'X_reduced': X_isomap,
                'transformer': isomap
            }
        except:
            print("      Isomap failed")
        
        # Locally Linear Embedding
        print("   Applying Locally Linear Embedding...")
        try:
            lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=min(10, X_sample.shape[0]-1))
            X_lle = lle.fit_transform(X_sample)
            
            results['lle'] = {
                'X_reduced': X_lle,
                'transformer': lle
            }
        except:
            print("      LLE failed")
        
        self.reduction_results['nonlinear'] = results
        return results
    
    def compare_reduction_methods(self, X, y, problem_type='classification'):
        """
        Compare dimensionality reduction methods by evaluating downstream performance
        """
        print("🏁 Comparing Dimensionality Reduction Methods")
        print("="*50)
        
        # Apply all reduction methods
        linear_results = self.apply_linear_reductions(X)
        nonlinear_results = self.apply_nonlinear_reductions(X)
        
        # Evaluation setup
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'r2'
        
        # Baseline performance
        baseline_scores = cross_val_score(estimator, X, y, cv=5, scoring=scoring)
        baseline_mean = np.mean(baseline_scores)
        
        comparison_results = {
            'baseline': {
                'score': baseline_mean,
                'std': np.std(baseline_scores),
                'features': X.shape[1]
            }
        }
        
        # Evaluate linear methods
        for method_name, method_results in linear_results.items():
            X_reduced = method_results['X_reduced']
            scores = cross_val_score(estimator, X_reduced, y, cv=5, scoring=scoring)
            
            comparison_results[method_name] = {
                'score': np.mean(scores),
                'std': np.std(scores),
                'features': X_reduced.shape[1]
            }
            
            print(f"   {method_name.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Evaluate nonlinear methods (if data was sampled, evaluate on sample)
        if len(nonlinear_results) > 0:
            print("\n   Nonlinear methods (evaluated on sample):")
            
            # Get sample indices if used
            if X.shape[0] > 1000:
                sample_indices = np.random.choice(X.shape[0], 1000, replace=False)
                y_sample = y[sample_indices]
            else:
                y_sample = y
            
            for method_name, method_results in nonlinear_results.items():
                X_reduced = method_results['X_reduced']
                scores = cross_val_score(estimator, X_reduced, y_sample, cv=5, scoring=scoring)
                
                comparison_results[f'{method_name}_sample'] = {
                    'score': np.mean(scores),
                    'std': np.std(scores),
                    'features': X_reduced.shape[1]
                }
                
                print(f"   {method_name.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Visualize comparison
        self._visualize_reduction_comparison(comparison_results)
        
        return comparison_results
    
    def _visualize_reduction_comparison(self, comparison_results):
        """
        Visualize dimensionality reduction comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        ax = axes[0]
        methods = list(comparison_results.keys())
        scores = [comparison_results[method]['score'] for method in methods]
        stds = [comparison_results[method]['std'] for method in methods]
        
        bars = ax.bar(range(len(methods)), scores, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Reduction Method')
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Performance Comparison')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Color bars by performance
        baseline_score = comparison_results['baseline']['score']
        for bar, score in zip(bars, scores):
            if score > baseline_score:
                bar.set_color('green')
            elif score < baseline_score - 0.02:
                bar.set_color('red')
            else:
                bar.set_color('orange')
        
        # Feature count comparison
        ax = axes[1]
        feature_counts = [comparison_results[method]['features'] for method in methods]
        
        bars = ax.bar(range(len(methods)), feature_counts, alpha=0.7, color='skyblue')
        ax.set_xlabel('Reduction Method')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Count Comparison')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# COMPREHENSIVE EXAMPLE AND DEMONSTRATIONS
# ==========================================

def run_comprehensive_feature_engineering_demo():
    """
    Comprehensive demonstration of all feature engineering techniques
    """
    print("🚀 Neural Learning Web - Week 23: Feature Engineering")
    print("="*80)
    
    # ================================================================
    # DATA PREPARATION
    # ================================================================
    
    print("\n📊 DATASET PREPARATION")
    print("="*50)
    
    # Create synthetic dataset for demonstration
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=3, n_clusters_per_class=1, random_state=42
    )
    
    print(f"Dataset: Synthetic Classification")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    # ================================================================
    # FEATURE CREATION FRAMEWORK
    # ================================================================
    
    print("\n🔢 FEATURE CREATION FRAMEWORK")
    print("="*50)
    
    feature_creator = FeatureCreationFramework()
    
    print("\n1️⃣ Mathematical Features")
    X_math, names_math = feature_creator.create_mathematical_features(X)
    
    print("\n2️⃣ Binning Features")
    X_binned, names_binned = feature_creator.create_binning_features(X_math, names_math)
    
    print("\n3️⃣ Clustering Features") 
    X_clustered, names_clustered = feature_creator.create_clustering_features(X_binned, names_binned)
    
    # ================================================================
    # ADVANCED FEATURE SELECTION
    # ================================================================
    
    print("\n🎯 ADVANCED FEATURE SELECTION")
    print("="*50)
    
    feature_selector = AdvancedFeatureSelection()
    
    print("\n1️⃣ Comprehensive Univariate Selection")
    univariate_results = feature_selector.univariate_selection_comprehensive(X_clustered, y)
    
    print("\n2️⃣ Multivariate Selection")
    multivariate_results = feature_selector.multivariate_selection(X_clustered, y)
    
    print("\n3️⃣ Stability Selection")
    stability_results = feature_selector.stability_selection(X_clustered, y)
    
    print("\n4️⃣ Selection Method Comparison")
    selection_comparison = feature_selector.compare_selection_methods(X_clustered, y)
    
    # ================================================================
    # AUTOMATED FEATURE ENGINEERING
    # ================================================================
    
    print("\n🤖 AUTOMATED FEATURE ENGINEERING PIPELINE")
    print("="*50)
    
    auto_engineer = AutomatedFeatureEngineering('classification')
    X_automated, names_automated, pipeline_results = auto_engineer.automated_pipeline(X, y)
    
    # ================================================================
    # FEATURE INTERACTIONS
    # ================================================================
    
    print("\n🔗 FEATURE INTERACTION ANALYSIS")
    print("="*50)
    
    interaction_framework = FeatureInteractionFramework()
    
    print("\n1️⃣ Detecting Feature Interactions")
    interactions = interaction_framework.detect_feature_interactions(X, y)
    
    print("\n2️⃣ Creating Polynomial Features")
    X_poly, names_poly = interaction_framework.create_polynomial_features(X, degree=2)
    
    print("\n3️⃣ Evaluating Interaction Impact")
    interaction_impact = interaction_framework.evaluate_interaction_impact(X, y, top_k_interactions=5)
    
    # ================================================================
    # DIMENSIONALITY REDUCTION
    # ================================================================
    
    print("\n📉 DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*50)
    
    dim_reducer = DimensionalityReductionFramework()
    reduction_comparison = dim_reducer.compare_reduction_methods(X_automated, y)
    
    # ================================================================
    # FINAL SUMMARY AND RECOMMENDATIONS
    # ================================================================
    
    print("\n🎯 FINAL FEATURE ENGINEERING SUMMARY")
    print("="*50)
    
    print(f"Pipeline Results:")
    print(f"  Original Features: {X.shape[1]}")
    print(f"  Final Features: {X_automated.shape[1]}")
    print(f"  Performance Improvement: {pipeline_results['improvement']:+.4f}")
    print(f"  Baseline Score: {pipeline_results['baseline_score']:.4f}")
    print(f"  Final Score: {pipeline_results['final_score']:.4f}")
    
    print(f"\nBest Dimensionality Reduction:")
    best_method = max(reduction_comparison.keys(), 
                     key=lambda k: reduction_comparison[k]['score'] if k != 'baseline' else -1)
    if best_method != 'baseline':
        print(f"  Method: {best_method}")
        print(f"  Score: {reduction_comparison[best_method]['score']:.4f}")
        print(f"  Features: {reduction_comparison[best_method]['features']}")
    
    print(f"\nFeature Selection Consensus:")
    consensus_features = selection_comparison['consensus_features']
    print(f"  Consensus Features: {len(consensus_features)}")
    print(f"  Feature Indices: {consensus_features[:10]}")  # Show first 10
    
    print(f"\nTop Feature Interactions:")
    for i, interaction in enumerate(interactions[:3]):
        print(f"  {i+1}. {interaction['feature1_name']} × {interaction['feature2_name']}: {interaction['score']:.4f}")
    
    return {
        'X_original': X,
        'X_final': X_automated,
        'pipeline_results': pipeline_results,
        'selection_results': selection_comparison,
        'interaction_results': interactions,
        'reduction_results': reduction_comparison
    }

# ==========================================
# SPECIALIZED FEATURE ENGINEERING EXERCISES
# ==========================================

def exercise_1_custom_transformations():
    """
    Exercise 1: Implement custom feature transformations
    """
    print("\n🎯 Exercise 1: Custom Feature Transformations")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = (X[:, 0] + X[:, 1]**2 + np.sin(X[:, 2]) + np.random.randn(500) * 0.1 > 0).astype(int)
    
    class CustomTransformer(BaseEstimator, TransformerMixin):
        """Custom transformer for domain-specific features"""
        
        def __init__(self, transform_type='all'):
            self.transform_type = transform_type
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X_transformed = X.copy()
            
            if self.transform_type in ['all', 'trigonometric']:
                # Add trigonometric features
                X_transformed = np.column_stack([
                    X_transformed,
                    np.sin(X),
                    np.cos(X)
                ])
            
            if self.transform_type in ['all', 'statistical']:
                # Add statistical features across rows
                row_stats = np.column_stack([
                    np.mean(X, axis=1),
                    np.std(X, axis=1),
                    np.min(X, axis=1),
                    np.max(X, axis=1)
                ])
                X_transformed = np.column_stack([X_transformed, row_stats])
            
            if self.transform_type in ['all', 'exponential']:
                # Add exponential and logarithmic features
                X_safe = np.abs(X) + 1e-8  # Ensure positive values
                X_transformed = np.column_stack([
                    X_transformed,
                    np.log(X_safe),
                    np.exp(np.clip(X, -2, 2))  # Clip to avoid overflow
                ])
            
            return X_transformed
    
    # Test different transformations
    transformers = {
        'original': None,
        'trigonometric': CustomTransformer('trigonometric'),
        'statistical': CustomTransformer('statistical'),
        'exponential': CustomTransformer('exponential'),
        'all': CustomTransformer('all')
    }
    
    results = {}
    
    for name, transformer in transformers.items():
        if transformer is None:
            X_trans = X
        else:
            X_trans = transformer.transform(X)
        
        # Evaluate performance
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(rf, X_trans, y, cv=5)
        
        results[name] = {
            'features': X_trans.shape[1],
            'score': np.mean(scores),
            'std': np.std(scores)
        }
        
        print(f"   {name}: {X_trans.shape[1]} features, Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return results

def exercise_2_time_series_features():
    """
    Exercise 2: Time series feature engineering
    """
    print("\n🎯 Exercise 2: Time Series Feature Engineering")
    print("="*50)
    
    # Generate synthetic time series
    np.random.seed(42)
    t = np.arange(100)
    trend = 0.01 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(100) * 0.5
    ts = trend + seasonal + noise
    
    # Create supervised learning problem (predict next value)
    def create_supervised_dataset(time_series, window_size=10):
        X, y = [], []
        for i in range(window_size, len(time_series)):
            X.append(time_series[i-window_size:i])
            y.append(time_series[i])
        return np.array(X), np.array(y)
    
    X_ts, y_ts = create_supervised_dataset(ts)
    
    class TimeSeriesFeatureExtractor:
        """Extract time series features from windows"""
        
        def __init__(self):
            pass
        
        def extract_features(self, X_windows):
            features = []
            
            for window in X_windows:
                # Statistical features
                stats = [
                    np.mean(window),
                    np.std(window),
                    np.min(window),
                    np.max(window),
                    np.ptp(window),  # peak-to-peak
                    np.median(window)
                ]
                
                # Trend features
                x = np.arange(len(window))
                slope = np.polyfit(x, window, 1)[0]
                
                # Seasonal features
                if len(window) >= 4:
                    fft = np.fft.fft(window)
                    dominant_freq = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                else:
                    dominant_freq = 0
                
                # Lag features
                lag1 = window[-1] - window[-2] if len(window) > 1 else 0
                lag2 = window[-1] - window[-3] if len(window) > 2 else 0
                
                # Moving averages
                ma3 = np.mean(window[-3:]) if len(window) >= 3 else np.mean(window)
                ma5 = np.mean(window[-5:]) if len(window) >= 5 else np.mean(window)
                
                window_features = stats + [slope, dominant_freq, lag1, lag2, ma3, ma5]
                features.append(window_features)
            
            return np.array(features)
    
    # Extract features
    feature_extractor = TimeSeriesFeatureExtractor()
    X_features = feature_extractor.extract_features(X_ts)
    
    print(f"   Original windows: {X_ts.shape}")
    print(f"   Extracted features: {X_features.shape}")
    
    # Compare performance
    models = {
        'Raw Windows': (X_ts, 'Using raw time series windows'),
        'Engineered Features': (X_features, 'Using engineered features')
    }
    
    for name, (X_model, description) in models.items():
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        scores = cross_val_score(rf, X_model, y_ts, cv=5, scoring='r2')
        print(f"   {name}: R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"      {description}")
    
    return X_features, y_ts

def exercise_3_categorical_encoding():
    """
    Exercise 3: Advanced categorical encoding techniques
    """
    print("\n🎯 Exercise 3: Advanced Categorical Encoding")
    print("="*50)
    
    # Create synthetic categorical data
    np.random.seed(42)
    n_samples = 1000
    
    # High cardinality categorical
    categories_high = ['cat_' + str(i) for i in range(50)]
    cat_high = np.random.choice(categories_high, n_samples)
    
    # Low cardinality categorical  
    categories_low = ['A', 'B', 'C', 'D']
    cat_low = np.random.choice(categories_low, n_samples)
    
    # Ordinal categorical
    cat_ordinal = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    
    # Numerical features
    X_num = np.random.randn(n_samples, 3)
    
    # Create target with relationship to categories
    target_mapping_high = {cat: np.random.randn() for cat in categories_high}
    target_mapping_low = {'A': -1, 'B': 0, 'C': 0.5, 'D': 1}
    target_mapping_ord = {'Low': -1, 'Medium': 0, 'High': 1}
    
    y = (np.array([target_mapping_high[c] for c in cat_high]) +
         np.array([target_mapping_low[c] for c in cat_low]) +
         np.array([target_mapping_ord[c] for c in cat_ordinal]) +
         np.sum(X_num, axis=1) +
         np.random.randn(n_samples) * 0.1)
    
    # Convert to binary classification
    y = (y > np.median(y)).astype(int)
    
    # Test different encoding strategies
    df = pd.DataFrame({
        'cat_high': cat_high,
        'cat_low': cat_low, 
        'cat_ordinal': cat_ordinal,
        'num1': X_num[:, 0],
        'num2': X_num[:, 1],
        'num3': X_num[:, 2]
    })
    
    encoding_strategies = {}
    
    # 1. One-Hot Encoding
    df_onehot = df.copy()
    df_onehot = pd.get_dummies(df_onehot, columns=['cat_low'], prefix='onehot')
    # Skip high cardinality for one-hot to avoid too many features
    df_onehot['cat_high_freq'] = df_onehot['cat_high'].map(df_onehot['cat_high'].value_counts())
    df_onehot = df_onehot.drop('cat_high', axis=1)
    # Manual ordinal encoding
    ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df_onehot['cat_ordinal'] = df_onehot['cat_ordinal'].map(ordinal_map)
    
    encoding_strategies['One-Hot'] = df_onehot.values
    
    # 2. Target Encoding
    df_target = df.copy()
    
    # Target encode high cardinality
    target_means_high = df_target.groupby('cat_high')['num1'].transform('mean')  # Use num1 as proxy
    df_target['cat_high_target'] = target_means_high
    df_target = df_target.drop('cat_high', axis=1)
    
    # Target encode low cardinality
    target_means_low = df_target.groupby('cat_low')['num2'].transform('mean')
    df_target['cat_low_target'] = target_means_low
    df_target = df_target.drop('cat_low', axis=1)
    
    # Ordinal encoding
    df_target['cat_ordinal'] = df_target['cat_ordinal'].map(ordinal_map)
    
    encoding_strategies['Target'] = df_target.values
    
    # 3. Frequency Encoding
    df_freq = df.copy()
    
    # Frequency encoding
    df_freq['cat_high_freq'] = df_freq['cat_high'].map(df_freq['cat_high'].value_counts())
    df_freq['cat_low_freq'] = df_freq['cat_low'].map(df_freq['cat_low'].value_counts())
    df_freq = df_freq.drop(['cat_high', 'cat_low'], axis=1)
    df_freq['cat_ordinal'] = df_freq['cat_ordinal'].map(ordinal_map)
    
    encoding_strategies['Frequency'] = df_freq.values
    
    # 4. Mixed Strategy
    df_mixed = df.copy()
    
    # One-hot for low cardinality
    df_mixed = pd.get_dummies(df_mixed, columns=['cat_low'], prefix='mixed')
    
    # Target encoding for high cardinality
    target_means = df_mixed.groupby('cat_high')['num1'].transform('mean')
    df_mixed['cat_high_target'] = target_means
    df_mixed = df_mixed.drop('cat_high', axis=1)
    
    # Ordinal for ordinal
    df_mixed['cat_ordinal'] = df_mixed['cat_ordinal'].map(ordinal_map)
    
    encoding_strategies['Mixed'] = df_mixed.values
    
    # Evaluate each strategy
    print("   Evaluating encoding strategies:")
    
    results = {}
    for strategy_name, X_encoded in encoding_strategies.items():
        # Handle any remaining string columns or NaN values
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.select_dtypes(include=[np.number]).fillna(0).values
        else:
            X_encoded = np.nan_to_num(X_encoded)
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(rf, X_encoded, y, cv=5)
        
        results[strategy_name] = {
            'score': np.mean(scores),
            'std': np.std(scores),
            'features': X_encoded.shape[1]
        }
        
        print(f"      {strategy_name}: {X_encoded.shape[1]} features, Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return results

# ==========================================
# MAIN EXECUTION FUNCTION
# ==========================================

if __name__ == "__main__":
    print("🧠 Neural Learning Web - Phase 2, Week 23")
    print("Feature Engineering - The Art and Science of Predictive Features")
    print("="*80)
    
    # Run comprehensive demonstration
    comprehensive_results = run_comprehensive_feature_engineering_demo()
    
    print("\n" + "="*80)
    print("📚 ADVANCED EXERCISES")
    print("="*80)
    
    # Run advanced exercises
    exercise_1_results = exercise_1_custom_transformations()
    exercise_2_results = exercise_2_time_series_features()
    exercise_3_results = exercise_3_categorical_encoding()
    
    print("\n" + "="*80)
    print("🎓 WEEK 23 MASTERY SUMMARY")
    print("="*80)
    
    print("✅ Feature Engineering Techniques Mastered:")
    print("   🔢 Mathematical feature transformations")
    print("   📊 Binning and discretization strategies")
    print("   🎯 Clustering-based feature creation")
    print("   🔍 Advanced univariate and multivariate selection")
    print("   🎪 Stability selection with bootstrap validation")
    print("   🤖 Automated feature engineering pipelines")
    print("   🔗 Feature interaction detection and creation")
    print("   📈 Polynomial feature generation")
    print("   📉 Linear and nonlinear dimensionality reduction")
    print("   📝 Domain-specific feature extraction")
    print("   ⏰ Time series feature engineering")
    print("   🏷️ Advanced categorical encoding strategies")
    
    print("\n🎯 Key Performance Insights:")
    pipeline_improvement = comprehensive_results['pipeline_results']['improvement']
    print(f"   📈 Automated pipeline improvement: {pipeline_improvement:+.4f}")
    print(f"   🎲 Feature selection consensus found: {len(comprehensive_results['selection_results']['consensus_features'])} features")
    print(f"   🔗 Top interaction score: {comprehensive_results['interaction_results'][0]['score']:.4f}")
    
    print("\n🚀 Ready for Advanced Topics:")
    print("   🏆 Week 24: Advanced ensemble methods")
    print("   🧠 Phase 3: Deep learning fundamentals")
    print("   🎨 Feature engineering will be crucial for:")
    print("      - Neural network input preprocessing")
    print("      - Transfer learning feature adaptation") 
    print("      - Multimodal data fusion")
    print("      - Production ML pipeline optimization")
    
    print("\n💡 Professional Applications:")
    print("   🏢 Industry: Custom feature engineering for domain expertise")
    print("   🔬 Research: Novel feature creation for competitive advantage")
    print("   📱 Products: Automated feature pipelines for production systems")
    print("   📊 Analytics: Feature engineering for business intelligence")
    
    print(f"\n🎉 Week 23 Complete! Feature Engineering Mastery Achieved!")
    print(f"Ready to combine these skills with advanced ensemble methods!")