"""
Neural Odyssey - Week 3: Probability and Statistics
Exercises for mastering the language of uncertainty in machine learning

This module implements core concepts that enable statistical reasoning:
- Probability theory and conditional reasoning
- Statistical distributions and pattern recognition
- Bayesian inference and belief updating
- Hypothesis testing and model evaluation

Complete the TODO functions to build your statistical reasoning toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: PROBABILITY FOUNDATIONS
# ==============================================

def calculate_probability(favorable_outcomes, total_outcomes):
    """
    Calculate basic probability: P(Event) = favorable / total
    
    This is the foundation of all probabilistic reasoning.
    """
    if total_outcomes == 0:
        return 0
    return favorable_outcomes / total_outcomes

def conditional_probability(p_a_and_b, p_b):
    """
    TODO: Calculate conditional probability P(A|B) = P(A ∩ B) / P(B)
    
    This is crucial for ML - how does observing B change our belief about A?
    Used everywhere: Naive Bayes, feature dependencies, causal reasoning.
    
    Args:
        p_a_and_b: Probability of both A and B occurring
        p_b: Probability of B occurring
        
    Returns:
        P(A|B) - probability of A given B
    """
    # TODO: Implement conditional probability
    # Handle the case where P(B) = 0
    
    pass

def independence_test(p_a, p_b, p_a_and_b, tolerance=1e-6):
    """
    TODO: Test if two events are independent
    
    Events A and B are independent if P(A ∩ B) = P(A) × P(B)
    This is the foundation of Naive Bayes assumption.
    
    Args:
        p_a: Probability of event A
        p_b: Probability of event B  
        p_a_and_b: Probability of both A and B
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Boolean indicating if events are independent
    """
    # TODO: Check if P(A ∩ B) ≈ P(A) × P(B)
    
    pass

def monte_carlo_probability(event_function, n_trials=10000, random_state=42):
    """
    TODO: Estimate probability using Monte Carlo simulation
    
    Sometimes analytical probability is hard to compute.
    Monte Carlo lets us estimate by simulation - very powerful!
    
    Args:
        event_function: Function that returns True if event occurs
        n_trials: Number of simulation trials
        random_state: For reproducibility
        
    Returns:
        Estimated probability of the event
    """
    np.random.seed(random_state)
    
    # TODO: Run n_trials simulations and count favorable outcomes
    # Return the fraction of trials where event_function() returns True
    
    pass

def birthday_paradox_simulation(n_people=23, n_trials=10000):
    """
    TODO: Simulate the famous birthday paradox
    
    What's the probability that in a group of n people,
    at least two share the same birthday?
    
    This demonstrates how counter-intuitive probability can be!
    
    Args:
        n_people: Size of the group
        n_trials: Number of simulation trials
        
    Returns:
        Estimated probability of shared birthday
    """
    # TODO: For each trial:
    # 1. Generate n_people random birthdays (1-365)
    # 2. Check if any two people share a birthday
    # 3. Count fraction of trials with shared birthdays
    
    pass

# ==============================================
# PART 2: STATISTICAL DISTRIBUTIONS
# ==============================================

def normal_distribution_pdf(x, mu=0, sigma=1):
    """
    TODO: Implement Normal distribution probability density function
    
    PDF(x) = (1/√(2πσ²)) * exp(-½((x-μ)/σ)²)
    
    The normal distribution appears everywhere due to Central Limit Theorem.
    Understanding it deeply is crucial for ML.
    """
    # TODO: Implement the normal PDF formula
    # Don't use scipy.stats - implement from scratch!
    
    pass

def binomial_distribution_pmf(k, n, p):
    """
    TODO: Implement Binomial distribution probability mass function
    
    PMF(k) = C(n,k) * p^k * (1-p)^(n-k)
    where C(n,k) = n! / (k!(n-k)!)
    
    Models number of successes in n independent trials.
    Foundation of logistic regression and A/B testing.
    """
    # TODO: Implement binomial PMF
    # Calculate combination C(n,k) and apply formula
    # Hint: Use math.factorial or implement combination function
    
    pass

def poisson_distribution_pmf(k, lambda_param):
    """
    TODO: Implement Poisson distribution probability mass function
    
    PMF(k) = (λ^k * e^(-λ)) / k!
    
    Models rare events and count data.
    Appears in web analytics, NLP, and recommendation systems.
    """
    # TODO: Implement Poisson PMF
    # Handle the case where k! could be large
    
    pass

def exponential_distribution_pdf(x, lambda_param):
    """
    TODO: Implement Exponential distribution PDF
    
    PDF(x) = λ * exp(-λx) for x ≥ 0
    
    Models waiting times and survival data.
    Used in reliability engineering and time-to-event analysis.
    """
    # TODO: Implement exponential PDF
    # Handle the case where x < 0
    
    pass

def central_limit_theorem_demo(population_dist, sample_size=30, n_samples=1000):
    """
    TODO: Demonstrate Central Limit Theorem empirically
    
    The CLT states that sample means approach normal distribution
    regardless of the population distribution shape.
    
    This is why we can assume normality for many statistical tests!
    
    Args:
        population_dist: Function that generates random samples from population
        sample_size: Size of each sample
        n_samples: Number of samples to take
        
    Returns:
        Array of sample means
    """
    # TODO: 
    # 1. Take n_samples from the population
    # 2. For each sample, calculate the mean
    # 3. Return array of sample means
    # 4. Plot histogram to show it's approximately normal
    
    pass

def distribution_identifier(data):
    """
    TODO: Identify which distribution best fits the data
    
    This is a crucial skill - recognizing distributional patterns
    helps choose appropriate models and preprocessing steps.
    
    Args:
        data: Array of observed values
        
    Returns:
        Dictionary with distribution name and parameters
    """
    # TODO: Test data against common distributions:
    # 1. Normal (check if roughly bell-shaped)
    # 2. Exponential (check if decay pattern)
    # 3. Uniform (check if roughly flat)
    # 4. Use statistical tests like Kolmogorov-Smirnov
    
    pass

# ==============================================
# PART 3: BAYESIAN REASONING
# ==============================================

def bayes_theorem(prior, likelihood, evidence):
    """
    TODO: Implement Bayes' theorem
    
    P(H|E) = P(E|H) * P(H) / P(E)
    
    This is the foundation of all Bayesian reasoning.
    """
    # TODO: Apply Bayes' theorem formula
    # Handle case where evidence = 0
    
    pass

def naive_bayes_classifier(features, labels, test_features):
    """
    TODO: Implement Naive Bayes classifier from scratch
    
    This is a direct application of Bayes' theorem with the
    "naive" assumption that features are independent.
    
    Steps:
    1. Calculate prior probabilities P(class)
    2. Calculate likelihoods P(feature|class) for each feature
    3. Apply Bayes' theorem: P(class|features) ∝ P(class) * ∏P(feature|class)
    
    Args:
        features: Training feature matrix (n_samples, n_features)
        labels: Training labels
        test_features: Test feature matrix
        
    Returns:
        Predicted class probabilities for test data
    """
    # TODO: Implement complete Naive Bayes
    # Handle both continuous (Gaussian) and discrete features
    
    pass

def bayesian_parameter_estimation(data, prior_mean=0, prior_variance=1):
    """
    TODO: Estimate parameters using Bayesian approach
    
    Instead of point estimates, Bayesian approach gives us
    full posterior distribution over parameters.
    
    For normal data with normal prior:
    Posterior is also normal with updated mean and variance.
    
    Args:
        data: Observed data points
        prior_mean: Prior belief about mean
        prior_variance: Prior uncertainty about mean
        
    Returns:
        Dictionary with posterior mean and variance
    """
    # TODO: Calculate Bayesian posterior for normal-normal model
    # Update prior beliefs based on observed data
    
    pass

def bayesian_ab_test(control_conversions, control_trials, 
                    test_conversions, test_trials, 
                    prior_alpha=1, prior_beta=1):
    """
    TODO: Implement Bayesian A/B testing
    
    Instead of p-values, use Bayesian approach to compare
    conversion rates with full uncertainty quantification.
    
    Use Beta-Binomial conjugate prior model:
    - Prior: Beta(α, β)
    - Likelihood: Binomial(n, p)
    - Posterior: Beta(α + successes, β + failures)
    
    Args:
        control_conversions: Number of conversions in control
        control_trials: Number of trials in control
        test_conversions: Number of conversions in test
        test_trials: Number of trials in test
        prior_alpha, prior_beta: Beta prior parameters
        
    Returns:
        Dictionary with posterior distributions and probability test > control
    """
    # TODO: 
    # 1. Calculate posterior Beta distributions for both groups
    # 2. Sample from posteriors to estimate P(test > control)
    # 3. Calculate credible intervals
    
    pass

def sequential_bayesian_updating(observations, prior_belief):
    """
    TODO: Demonstrate sequential Bayesian updating
    
    Show how beliefs evolve as we see more data.
    Each observation updates our posterior, which becomes
    the prior for the next observation.
    
    Args:
        observations: Sequence of binary observations (0 or 1)
        prior_belief: Initial Beta prior (alpha, beta)
        
    Returns:
        List of posterior beliefs after each observation
    """
    # TODO: Update beliefs sequentially using Beta-Binomial model
    
    pass

# ==============================================
# PART 4: HYPOTHESIS TESTING AND INFERENCE
# ==============================================

def t_test_one_sample(data, null_mean=0, alpha=0.05):
    """
    TODO: Implement one-sample t-test from scratch
    
    Tests if sample mean significantly differs from null hypothesis.
    Foundation of statistical significance testing in ML.
    
    Steps:
    1. Calculate sample mean and standard error
    2. Compute t-statistic: (sample_mean - null_mean) / standard_error
    3. Find p-value using t-distribution
    4. Compare p-value to significance level α
    
    Args:
        data: Sample data
        null_mean: Hypothesized population mean
        alpha: Significance level
        
    Returns:
        Dictionary with t-statistic, p-value, and decision
    """
    # TODO: Implement complete t-test
    # Calculate degrees of freedom = n - 1
    # Use two-tailed test
    
    pass

def t_test_two_sample(sample1, sample2, alpha=0.05):
    """
    TODO: Implement two-sample t-test (Welch's t-test)
    
    Tests if two samples have significantly different means.
    Used for comparing ML model performance.
    
    Args:
        sample1, sample2: Two independent samples
        alpha: Significance level
        
    Returns:
        Dictionary with t-statistic, p-value, and decision
    """
    # TODO: Implement Welch's t-test for unequal variances
    # Calculate pooled standard error
    # Use Welch-Satterthwaite equation for degrees of freedom
    
    pass

def bootstrap_confidence_interval(data, statistic_function, confidence_level=0.95, n_bootstrap=1000):
    """
    TODO: Calculate confidence interval using bootstrap resampling
    
    Bootstrap is a powerful non-parametric method for
    estimating sampling distributions and confidence intervals.
    
    Steps:
    1. Resample data with replacement many times
    2. Calculate statistic for each bootstrap sample
    3. Use percentiles of bootstrap distribution for CI
    
    Args:
        data: Original sample
        statistic_function: Function to compute statistic (e.g., np.mean)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        
    Returns:
        Tuple (lower_bound, upper_bound) of confidence interval
    """
    # TODO: Implement bootstrap confidence interval
    # Use percentile method for CI bounds
    
    pass

def permutation_test(group1, group2, n_permutations=1000, random_state=42):
    """
    TODO: Implement permutation test for comparing groups
    
    Non-parametric test that doesn't assume specific distributions.
    Tests null hypothesis that groups come from same distribution.
    
    Steps:
    1. Calculate observed difference between groups
    2. Pool all data and randomly reassign to groups
    3. Calculate difference for each permutation
    4. p-value = fraction of permutations with difference ≥ observed
    
    Args:
        group1, group2: Data from two groups
        n_permutations: Number of random permutations
        random_state: For reproducibility
        
    Returns:
        Dictionary with observed difference, p-value, and conclusion
    """
    # TODO: Implement complete permutation test
    
    pass

def multiple_testing_correction(p_values, method='bonferroni'):
    """
    TODO: Apply multiple testing correction
    
    When testing many hypotheses, we need to control
    family-wise error rate or false discovery rate.
    
    Args:
        p_values: List of p-values from multiple tests
        method: 'bonferroni', 'holm', or 'benjamini_hochberg'
        
    Returns:
        Corrected p-values
    """
    # TODO: Implement multiple testing corrections
    # Bonferroni: multiply by number of tests
    # Holm: step-down method
    # Benjamini-Hochberg: controls false discovery rate
    
    pass

# ==============================================
# PART 5: STATISTICAL ANALYSIS TOOLKIT
# ==============================================

class StatisticalAnalyzer:
    """
    TODO: Build comprehensive statistical analysis toolkit
    
    This class should provide all tools needed for statistical
    analysis in ML projects:
    - Descriptive statistics
    - Distribution fitting
    - Hypothesis testing
    - Confidence intervals
    - Bayesian analysis
    """
    
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
        
    def descriptive_statistics(self):
        """
        TODO: Calculate comprehensive descriptive statistics
        
        Returns:
            Dictionary with mean, median, mode, std, skewness, kurtosis, etc.
        """
        # TODO: Calculate all relevant descriptive statistics
        
        pass
    
    def fit_distributions(self):
        """
        TODO: Fit multiple distributions and find best fit
        
        Returns:
            Dictionary with fitted distributions and goodness-of-fit measures
        """
        # TODO: Fit normal, exponential, gamma, etc.
        # Use AIC/BIC for model selection
        
        pass
    
    def outlier_detection(self, method='iqr'):
        """
        TODO: Detect outliers using statistical methods
        
        Args:
            method: 'iqr', 'zscore', or 'modified_zscore'
            
        Returns:
            Boolean array indicating outliers
        """
        # TODO: Implement multiple outlier detection methods
        
        pass
    
    def correlation_analysis(self, other_data):
        """
        TODO: Comprehensive correlation analysis
        
        Args:
            other_data: Another dataset to correlate with
            
        Returns:
            Dictionary with Pearson, Spearman, Kendall correlations
        """
        # TODO: Calculate multiple correlation coefficients
        # Include confidence intervals and significance tests
        
        pass
    
    def generate_report(self):
        """
        TODO: Generate comprehensive statistical report
        
        Returns:
            Formatted string with statistical summary
        """
        # TODO: Create human-readable statistical report
        
        pass

# ==============================================
# PART 6: REAL-WORLD APPLICATIONS
# ==============================================

def analyze_ab_test_results(control_data, test_data, metric='conversion_rate'):
    """
    TODO: Complete A/B test analysis framework
    
    Combines frequentist and Bayesian approaches for
    comprehensive analysis of experimental results.
    """
    # TODO: Implement complete A/B test analysis
    
    pass

def medical_diagnosis_bayesian(symptoms, disease_prevalence, symptom_probabilities):
    """
    TODO: Implement medical diagnosis using Bayes' theorem
    
    Classic application showing power of Bayesian reasoning
    in handling uncertainty and incorporating prior knowledge.
    """
    # TODO: Calculate posterior probability of disease given symptoms
    
    pass

def quality_control_analysis(measurements, spec_limits):
    """
    TODO: Statistical quality control analysis
    
    Use statistical methods to monitor manufacturing processes
    and detect when they go out of control.
    """
    # TODO: Implement control charts and process capability analysis
    
    pass

def financial_risk_assessment(returns_data, confidence_level=0.95):
    """
    TODO: Value at Risk (VaR) calculation using different methods
    
    Estimate potential losses with given confidence level
    using parametric, historical, and Monte Carlo methods.
    """
    # TODO: Calculate VaR using multiple approaches
    
    pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_probability_basics():
    """Demonstrate fundamental probability concepts."""
    
    print("🎯 Probability Foundations Demonstration")
    print("=" * 50)
    
    print("Basic probability calculation:")
    prob = calculate_probability(favorable_outcomes=2, total_outcomes=6)
    print(f"Rolling a die, P(getting 1 or 2) = {prob:.3f}")
    
    try:
        # Test conditional probability
        print("\nConditional probability example:")
        # P(A|B) where A="rain tomorrow", B="cloudy today"
        p_rain_and_cloudy = 0.3
        p_cloudy = 0.4
        p_rain_given_cloudy = conditional_probability(p_rain_and_cloudy, p_cloudy)
        print(f"P(rain tomorrow | cloudy today) = {p_rain_given_cloudy:.3f}")
        
        # Test independence
        p_a, p_b = 0.3, 0.4
        p_a_and_b_indep = 0.12  # 0.3 * 0.4
        p_a_and_b_dep = 0.20    # Not equal to product
        
        indep1 = independence_test(p_a, p_b, p_a_and_b_indep)
        indep2 = independence_test(p_a, p_b, p_a_and_b_dep)
        print(f"Independent events test: {indep1}")
        print(f"Dependent events test: {indep2}")
        
        # Birthday paradox
        birthday_prob = birthday_paradox_simulation(n_people=23)
        print(f"\nBirthday paradox (23 people): {birthday_prob:.3f}")
        
    except Exception as e:
        print(f"❌ Probability demo failed: {e}")
        print("Implement the conditional_probability() and related functions!")

def demonstrate_distributions():
    """Demonstrate statistical distributions and CLT."""
    
    print("\n📊 Statistical Distributions Demonstration")
    print("=" * 50)
    
    # Test distribution functions
    try:
        # Normal distribution
        x_vals = np.linspace(-3, 3, 100)
        normal_probs = [normal_distribution_pdf(x) for x in x_vals]
        print("✅ Normal distribution PDF implemented")
        
        # Binomial distribution
        binom_prob = binomial_distribution_pmf(k=7, n=10, p=0.6)
        print(f"Binomial P(7 successes in 10 trials, p=0.6) = {binom_prob:.4f}")
        
        # Central Limit Theorem demo
        def uniform_population():
            return np.random.uniform(0, 1, 100)
        
        sample_means = central_limit_theorem_demo(uniform_population)
        print("✅ Central Limit Theorem demonstration completed")
        print("Sample means from uniform population approach normal distribution!")
        
    except Exception as e:
        print(f"❌ Distributions demo failed: {e}")
        print("Implement the distribution functions!")

def demonstrate_bayesian_reasoning():
    """Demonstrate Bayesian inference applications."""
    
    print("\n🧠 Bayesian Reasoning Demonstration")
    print("=" * 50)
    
    try:
        # Bayes' theorem example
        prior = 0.01  # Disease prevalence
        likelihood = 0.95  # Test sensitivity
        evidence = 0.059  # P(positive test)
        posterior = bayes_theorem(prior, likelihood, evidence)
        print(f"Medical diagnosis: P(disease | positive test) = {posterior:.3f}")
        
        # Naive Bayes classifier demo
        # Simple 2D dataset
        features = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        test_features = np.array([[1.5, 1.5], [3.5, 3.5]])
        
        predictions = naive_bayes_classifier(features, labels, test_features)
        print(f"✅ Naive Bayes predictions: {predictions}")
        
        # Bayesian A/B test
        ab_results = bayesian_ab_test(
            control_conversions=120, control_trials=1000,
            test_conversions=140, test_trials=1000
        )
        print(f"✅ Bayesian A/B test completed")
        
    except Exception as e:
        print(f"❌ Bayesian demo failed: {e}")
        print("Implement the Bayesian functions!")

def demonstrate_hypothesis_testing():
    """Demonstrate statistical hypothesis testing."""
    
    print("\n🔬 Hypothesis Testing Demonstration")
    print("=" * 50)
    
    try:
        # Generate sample data
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 50)
        sample2 = np.random.normal(105, 15, 50)
        
        # One-sample t-test
        t_result = t_test_one_sample(sample1, null_mean=100)
        print(f"One-sample t-test: t={t_result['t_statistic']:.3f}, p={t_result['p_value']:.3f}")
        
        # Two-sample t-test
        t2_result = t_test_two_sample(sample1, sample2)
        print(f"Two-sample t-test: t={t2_result['t_statistic']:.3f}, p={t2_result['p_value']:.3f}")
        
        # Bootstrap confidence interval
        ci = bootstrap_confidence_interval(sample1, np.mean)
        print(f"Bootstrap 95% CI for mean: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        # Permutation test
        perm_result = permutation_test(sample1, sample2)
        print(f"Permutation test p-value: {perm_result['p_value']:.3f}")
        
    except Exception as e:
        print(f"❌ Hypothesis testing demo failed: {e}")
        print("Implement the statistical test functions!")

def comprehensive_statistical_analysis():
    """Run complete statistical analysis on real-like data."""
    
    print("\n🚀 Comprehensive Statistical Analysis")
    print("=" * 50)
    
    # Generate realistic dataset
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(50, 10, 800),    # Main population
        np.random.normal(80, 5, 150),     # Subpopulation
        np.random.uniform(20, 30, 50)     # Outliers
    ])
    
    try:
        # Create analyzer
        analyzer = StatisticalAnalyzer(data)
        
        # Run comprehensive analysis
        desc_stats = analyzer.descriptive_statistics()
        print(f"✅ Descriptive statistics calculated")
        
        dist_fits = analyzer.fit_distributions()
        print(f"✅ Distribution fitting completed")
        
        outliers = analyzer.outlier_detection()
        print(f"✅ Outlier detection: {np.sum(outliers)} outliers found")
        
        report = analyzer.generate_report()
        print(f"✅ Statistical report generated")
        
        print("\n🎉 Congratulations! You've built a complete statistical analysis toolkit!")
        print("This foundation will power all your machine learning work.")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("Implement the StatisticalAnalyzer class!")

if __name__ == "__main__":
    """
    Run this file to explore probability and statistics for ML!
    
    Complete the TODO functions above, then run:
    python week3_exercises.py
    """
    
    print("📊 Welcome to Neural Odyssey Week 3: Probability and Statistics!")
    print("Complete the TODO functions to master the language of uncertainty.")
    print("\nTo get started:")
    print("1. Implement basic probability functions (conditional_probability, etc.)")
    print("2. Build distribution functions from scratch")
    print("3. Master Bayesian reasoning with practical applications")
    print("4. Implement statistical tests for ML model evaluation")
    print("5. Create the comprehensive StatisticalAnalyzer toolkit")
    
    # Uncomment these lines after implementing the functions:
    # demonstrate_probability_basics()
    # demonstrate_distributions()
    # demonstrate_bayesian_reasoning()
    # demonstrate_hypothesis_testing()
    # comprehensive_statistical_analysis()
    
    print("\n💡 Pro tip: Statistics is the foundation of machine learning!")
    print("Master uncertainty quantification and you'll understand why ML works.")
    
    print("\n🎯 Success metrics:")
    print("• Can you explain Bayes' theorem and apply it to real problems?")
    print("• Can you identify distributions in data and understand their implications?")
    print("• Can you design statistical tests to compare ML models?")
    print("• Can you quantify uncertainty in predictions?")
    
    print("\n🏆 Master this week and you'll think like a true data scientist!")
    print("Probability is the language of uncertainty - and uncertainty is everywhere in data!")