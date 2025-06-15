#!/usr/bin/env python3
"""
Neural Odyssey - Week 31: Reinforcement Learning
Phase 2: Core ML Algorithms

Comprehensive exercises for understanding reinforcement learning from first principles.
This week you'll implement Q-learning, policy gradients, and explore the foundations
of decision-making under uncertainty.

Key Learning Objectives:
1. Understand Markov Decision Processes (MDPs)
2. Implement value-based methods (Q-learning, SARSA)
3. Build policy-based methods (REINFORCE, Actor-Critic)
4. Explore multi-armed bandits and exploration strategies
5. Apply RL to game environments and control problems
6. Understand the exploration vs exploitation trade-off
7. Implement experience replay and target networks

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import random
from typing import Tuple, List, Dict, Optional, Any, Union
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GridWorld:
    """
    Simple Grid World environment for RL experiments
    
    A classic RL environment where an agent navigates a grid to reach goals
    while avoiding obstacles. Perfect for understanding fundamental RL concepts.
    """
    
    def __init__(self, width: int = 5, height: int = 5, 
                 goal_reward: float = 10.0, step_penalty: float = -0.1,
                 obstacle_penalty: float = -1.0):
        """
        Initialize Grid World
        
        Args:
            width: Grid width
            height: Grid height
            goal_reward: Reward for reaching goal
            step_penalty: Small penalty for each step
            obstacle_penalty: Penalty for hitting obstacles
        """
        self.width = width
        self.height = height
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        
        # Define state space
        self.n_states = width * height
        self.n_actions = 4  # up, down, left, right
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Initialize environment
        self.reset()
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup goals and obstacles"""
        # Goal state (top-right corner)
        self.goal_state = (0, self.width - 1)
        
        # Obstacles (simple pattern)
        self.obstacles = {
            (1, 1), (1, 2), (2, 3), (3, 1)
        }
        
        # Remove obstacles that conflict with start/goal
        self.obstacles.discard((self.height - 1, 0))  # start
        self.obstacles.discard(self.goal_state)       # goal
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to starting state"""
        self.agent_pos = (self.height - 1, 0)  # bottom-left corner
        self.done = False
        self.steps = 0
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take an action in the environment
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            next_state, reward, done, info
        """
        if self.done:
            return self.agent_pos, 0, True, {}
        
        # Calculate new position
        dy, dx = self.actions[action]
        new_y = max(0, min(self.height - 1, self.agent_pos[0] + dy))
        new_x = max(0, min(self.width - 1, self.agent_pos[1] + dx))
        new_pos = (new_y, new_x)
        
        # Calculate reward
        reward = self.step_penalty
        
        # Check for obstacles
        if new_pos in self.obstacles:
            reward += self.obstacle_penalty
            # Don't move into obstacle
            new_pos = self.agent_pos
        
        # Check for goal
        if new_pos == self.goal_state:
            reward += self.goal_reward
            self.done = True
        
        # Update state
        self.agent_pos = new_pos
        self.steps += 1
        
        # Episode timeout
        if self.steps > 100:
            self.done = True
        
        return self.agent_pos, reward, self.done, {'steps': self.steps}
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert 2D state to 1D index"""
        return state[0] * self.width + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert 1D index to 2D state"""
        return (index // self.width, index % self.width)
    
    def render(self, q_values: Optional[np.ndarray] = None):
        """Visualize the grid world"""
        plt.figure(figsize=(10, 8))
        
        # Create grid visualization
        grid = np.zeros((self.height, self.width))
        
        # Mark special states
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == self.agent_pos:
                    grid[y, x] = 3  # Agent
                elif (y, x) == self.goal_state:
                    grid[y, x] = 2  # Goal
                elif (y, x) in self.obstacles:
                    grid[y, x] = -1  # Obstacle
                else:
                    grid[y, x] = 0  # Empty
        
        # Plot grid
        plt.subplot(1, 2, 1)
        plt.imshow(grid, cmap='RdYlBu', origin='upper')
        
        # Add labels
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == self.agent_pos:
                    plt.text(x, y, 'A', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (y, x) == self.goal_state:
                    plt.text(x, y, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (y, x) in self.obstacles:
                    plt.text(x, y, 'X', ha='center', va='center', fontsize=16, fontweight='bold')
        
        plt.title('Grid World Environment')
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        
        # Plot Q-values if provided
        if q_values is not None:
            plt.subplot(1, 2, 2)
            # Show value function (max Q-value for each state)
            value_grid = np.zeros((self.height, self.width))
            for y in range(self.height):
                for x in range(self.width):
                    state_idx = self.state_to_index((y, x))
                    value_grid[y, x] = np.max(q_values[state_idx])
            
            im = plt.imshow(value_grid, cmap='viridis', origin='upper')
            plt.colorbar(im)
            plt.title('Value Function')
            plt.xticks(range(self.width))
            plt.yticks(range(self.height))
            
            # Add arrows showing policy
            for y in range(self.height):
                for x in range(self.width):
                    if (y, x) not in self.obstacles and (y, x) != self.goal_state:
                        state_idx = self.state_to_index((y, x))
                        best_action = np.argmax(q_values[state_idx])
                        dy, dx = self.actions[best_action]
                        plt.arrow(x, y, dx*0.3, dy*0.3, head_width=0.1, 
                                head_length=0.1, fc='white', ec='white')
        
        plt.tight_layout()
        plt.show()


class MultiArmedBandit:
    """
    Multi-Armed Bandit environment
    
    Classic RL problem for understanding exploration vs exploitation trade-off.
    Each arm has a different reward distribution.
    """
    
    def __init__(self, n_arms: int = 10, reward_std: float = 1.0):
        """
        Initialize Multi-Armed Bandit
        
        Args:
            n_arms: Number of bandit arms
            reward_std: Standard deviation of reward noise
        """
        self.n_arms = n_arms
        self.reward_std = reward_std
        
        # Each arm has a different true mean reward
        self.true_means = np.random.normal(0, 1, n_arms)
        self.optimal_arm = np.argmax(self.true_means)
        
        # Statistics
        self.arm_counts = np.zeros(n_arms)
        self.total_reward = 0
        self.total_pulls = 0
    
    def pull_arm(self, arm: int) -> float:
        """
        Pull an arm and get reward
        
        Args:
            arm: Arm index to pull
            
        Returns:
            Reward from the arm
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm {arm}. Must be 0-{self.n_arms-1}")
        
        # Generate reward from arm's distribution
        reward = np.random.normal(self.true_means[arm], self.reward_std)
        
        # Update statistics
        self.arm_counts[arm] += 1
        self.total_reward += reward
        self.total_pulls += 1
        
        return reward
    
    def get_regret(self) -> float:
        """Calculate cumulative regret"""
        optimal_reward = self.true_means[self.optimal_arm] * self.total_pulls
        return optimal_reward - self.total_reward
    
    def reset(self):
        """Reset bandit statistics"""
        self.arm_counts = np.zeros(self.n_arms)
        self.total_reward = 0
        self.total_pulls = 0


class QLearningAgent:
    """
    Q-Learning Agent implementation
    
    Implements the classic Q-learning algorithm for model-free RL.
    Uses temporal difference learning to estimate action-value function.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        """
        Initialize Q-Learning Agent
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'epsilon': []
        }
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses exploration)
            
        Returns:
            Action to take
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update Q-table using Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def train_episode(self, env) -> Tuple[float, int]:
        """
        Train for one episode
        
        Args:
            env: Environment to train on
            
        Returns:
            Episode reward and length
        """
        state = env.reset()
        if hasattr(env, 'state_to_index'):
            state = env.state_to_index(state)
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Choose action
            action = self.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            if hasattr(env, 'state_to_index'):
                next_state = env.state_to_index(next_state)
            
            # Update Q-table
            self.update(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        return episode_reward, episode_length
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> Dict:
        """
        Train the Q-learning agent
        
        Args:
            env: Environment to train on
            n_episodes: Number of episodes to train
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        print(f"🤖 Training Q-Learning Agent")
        print(f"   Episodes: {n_episodes}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Discount factor: {self.discount_factor}")
        print(f"   Initial epsilon: {self.epsilon}")
        
        for episode in range(n_episodes):
            episode_reward, episode_length = self.train_episode(env)
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Store training history
            if episode % 10 == 0:
                self.training_history['episodes'].append(episode)
                self.training_history['rewards'].append(np.mean(self.episode_rewards[-10:]))
                self.training_history['lengths'].append(np.mean(self.episode_lengths[-10:]))
                self.training_history['epsilon'].append(self.epsilon)
            
            # Print progress
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"   Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.2f}, Epsilon = {self.epsilon:.3f}")
        
        return self.training_history


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) Agent
    
    On-policy temporal difference learning algorithm.
    Updates Q-values based on the actual policy being followed.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        """Initialize SARSA Agent"""
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, done: bool):
        """
        SARSA update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * self.q_table[next_state, next_action]
        
        td_error = target_q - current_q
        self.q_table[state, action] += self.learning_rate * td_error
    
    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode using SARSA"""
        state = env.reset()
        if hasattr(env, 'state_to_index'):
            state = env.state_to_index(state)
        
        action = self.choose_action(state)
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Take action
            next_state, reward, done, _ = env.step(action)
            if hasattr(env, 'state_to_index'):
                next_state = env.state_to_index(next_state)
            
            # Choose next action
            next_action = self.choose_action(next_state) if not done else 0
            
            # SARSA update
            self.update(state, action, reward, next_state, next_action, done)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            # Move to next state-action pair
            state = next_state
            action = next_action
        
        return episode_reward, episode_length


class PolicyGradientAgent:
    """
    REINFORCE Policy Gradient Agent
    
    Learns a parameterized policy directly using policy gradients.
    Uses the likelihood ratio method for gradient estimation.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.01,
                 discount_factor: float = 0.95):
        """
        Initialize Policy Gradient Agent
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate
            discount_factor: Discount factor
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Policy parameters (simple linear policy)
        self.policy_weights = np.random.normal(0, 0.1, (n_states, n_actions))
        
        # Statistics
        self.episode_rewards = []
        self.training_history = {'episodes': [], 'rewards': []}
    
    def softmax_policy(self, state: int) -> np.ndarray:
        """
        Compute softmax policy for given state
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        logits = self.policy_weights[state]
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def choose_action(self, state: int) -> int:
        """
        Sample action from policy
        
        Args:
            state: Current state
            
        Returns:
            Sampled action
        """
        action_probs = self.softmax_policy(state)
        return np.random.choice(self.n_actions, p=action_probs)
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted returns for episode
        
        Args:
            rewards: List of rewards from episode
            
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.append(G)
        
        returns.reverse()
        return returns
    
    def update_policy(self, states: List[int], actions: List[int], returns: List[float]):
        """
        Update policy using REINFORCE algorithm
        
        Args:
            states: States visited in episode
            actions: Actions taken in episode
            returns: Discounted returns for each step
        """
        # Normalize returns for stability
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Policy gradient update
        for state, action, G in zip(states, actions, returns):
            # Get action probabilities
            action_probs = self.softmax_policy(state)
            
            # Compute gradient of log policy
            grad_log_policy = np.zeros(self.n_actions)
            grad_log_policy[action] = 1.0 - action_probs[action]
            for a in range(self.n_actions):
                if a != action:
                    grad_log_policy[a] = -action_probs[a]
            
            # Policy gradient update
            self.policy_weights[state] += self.learning_rate * G * grad_log_policy
    
    def train_episode(self, env) -> float:
        """
        Train for one episode using REINFORCE
        
        Args:
            env: Environment to train on
            
        Returns:
            Episode reward
        """
        state = env.reset()
        if hasattr(env, 'state_to_index'):
            state = env.state_to_index(state)
        
        states, actions, rewards = [], [], []
        
        while True:
            # Choose action
            action = self.choose_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            if hasattr(env, 'state_to_index'):
                next_state = env.state_to_index(next_state)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # Compute returns and update policy
        returns = self.compute_returns(rewards)
        self.update_policy(states, actions, returns)
        
        episode_reward = sum(rewards)
        return episode_reward
    
    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> Dict:
        """
        Train the policy gradient agent
        
        Args:
            env: Environment to train on
            n_episodes: Number of episodes to train
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        print(f"🎭 Training Policy Gradient Agent")
        print(f"   Episodes: {n_episodes}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Discount factor: {self.discount_factor}")
        
        for episode in range(n_episodes):
            episode_reward = self.train_episode(env)
            self.episode_rewards.append(episode_reward)
            
            # Store training history
            if episode % 10 == 0:
                self.training_history['episodes'].append(episode)
                self.training_history['rewards'].append(np.mean(self.episode_rewards[-10:]))
            
            # Print progress
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"   Episode {episode}: Avg Reward = {avg_reward:.2f}")
        
        return self.training_history


class BanditAgent:
    """
    Multi-Armed Bandit Agent with different exploration strategies
    
    Implements various algorithms for solving the multi-armed bandit problem:
    - Epsilon-greedy
    - Upper Confidence Bound (UCB)
    - Thompson Sampling
    """
    
    def __init__(self, n_arms: int, strategy: str = 'epsilon_greedy', **kwargs):
        """
        Initialize Bandit Agent
        
        Args:
            n_arms: Number of bandit arms
            strategy: Exploration strategy ('epsilon_greedy', 'ucb', 'thompson')
            **kwargs: Strategy-specific parameters
        """
        self.n_arms = n_arms
        self.strategy = strategy
        
        # Strategy parameters
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.c = kwargs.get('c', 1.0)  # UCB confidence parameter
        
        # Statistics
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_reward = 0
        self.regrets = []
        
        # Thompson Sampling parameters (Beta distribution)
        self.alpha = np.ones(n_arms)  # Successes
        self.beta = np.ones(n_arms)   # Failures
    
    def choose_arm(self, t: int) -> int:
        """
        Choose arm based on strategy
        
        Args:
            t: Current time step
            
        Returns:
            Arm to pull
        """
        if self.strategy == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.strategy == 'ucb':
            return self._ucb(t)
        elif self.strategy == 'thompson':
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _epsilon_greedy(self) -> int:
        """Epsilon-greedy arm selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            # Handle case where no arms have been pulled
            avg_rewards = np.divide(self.arm_rewards, self.arm_counts, 
                                  out=np.zeros_like(self.arm_rewards), 
                                  where=self.arm_counts!=0)
            return np.argmax(avg_rewards)
    
    def _ucb(self, t: int) -> int:
        """Upper Confidence Bound arm selection"""
        if t < self.n_arms:
            # Pull each arm once initially
            return t
        
        avg_rewards = self.arm_rewards / np.maximum(self.arm_counts, 1)
        confidence_bounds = self.c * np.sqrt(np.log(t) / np.maximum(self.arm_counts, 1))
        ucb_values = avg_rewards + confidence_bounds
        
        return np.argmax(ucb_values)
    
    def _thompson_sampling(self) -> int:
        """Thompson Sampling arm selection"""
        # Sample from posterior Beta distributions
        sampled_means = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_means)
    
    def update(self, arm: int, reward: float):
        """
        Update statistics after pulling arm
        
        Args:
            arm: Arm that was pulled
            reward: Reward received
        """
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_reward += reward
        
        # Update Thompson Sampling parameters
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def train(self, bandit: MultiArmedBandit, n_steps: int = 1000) -> Dict:
        """
        Train on multi-armed bandit
        
        Args:
            bandit: Bandit environment
            n_steps: Number of steps to train
            
        Returns:
            Training history
        """
        rewards = []
        regrets = []
        
        for t in range(n_steps):
            # Choose arm
            arm = self.choose_arm(t)
            
            # Pull arm and get reward
            reward = bandit.pull_arm(arm)
            
            # Update agent
            self.update(arm, reward)
            
            # Calculate regret
            regret = bandit.get_regret()
            
            rewards.append(reward)
            regrets.append(regret)
        
        return {
            'rewards': rewards,
            'regrets': regrets,
            'cumulative_rewards': np.cumsum(rewards),
            'arm_counts': self.arm_counts.copy()
        }


# ==========================================
# EXERCISE IMPLEMENTATIONS
# ==========================================

def exercise_1_q_learning_gridworld():
    """
    Exercise 1: Implement Q-Learning in Grid World
    
    Learn the fundamentals of value-based RL by training a Q-learning
    agent to navigate a grid world environment.
    """
    print("\n🎯 Exercise 1: Q-Learning in Grid World")
    print("=" * 60)
    
       # Create environment
    env = GridWorld(width=5, height=5, goal_reward=10.0, step_penalty=-0.1)
    print(f"Environment: {env.width}x{env.height} Grid World")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    
    # Show initial environment
    print("\nInitial Environment:")
    env.render()
    
    # Create Q-learning agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    # Train agent
    print("\nTraining Q-Learning Agent...")
    history = agent.train(env, n_episodes=500, verbose=True)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    plt.plot(history['episodes'], history['rewards'], linewidth=2)
    plt.title('Average Reward vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['episodes'], history['lengths'], linewidth=2, color='orange')
    plt.title('Average Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Average Length')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['episodes'], history['epsilon'], linewidth=2, color='red')
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Q-table analysis
    plt.subplot(2, 3, 4)
    q_values_reshaped = agent.q_table.reshape(env.height, env.width, env.n_actions)
    max_q_values = np.max(q_values_reshaped, axis=2)
    im = plt.imshow(max_q_values, cmap='viridis')
    plt.colorbar(im)
    plt.title('Learned Value Function')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Policy visualization
    plt.subplot(2, 3, 5)
    policy_grid = np.argmax(q_values_reshaped, axis=2)
    plt.imshow(policy_grid, cmap='tab10')
    plt.title('Learned Policy')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Add policy arrows
    action_symbols = ['↑', '↓', '←', '→']
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) not in env.obstacles and (y, x) != env.goal_state:
                action = policy_grid[y, x]
                plt.text(x, y, action_symbols[action], ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='white')
    
    # Convergence analysis
    plt.subplot(2, 3, 6)
    # Calculate Q-table change over time (if tracked)
    final_rewards = agent.episode_rewards[-100:]
    plt.hist(final_rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Final 100 Episode Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Q-Learning Results', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Test learned policy
    print("\nTesting Learned Policy:")
    env.reset()
    env.render(agent.q_table)
    
    # Evaluate performance
    test_rewards = []
    test_lengths = []
    
    for _ in range(100):
        state = env.reset()
        state = env.state_to_index(state)
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.choose_action(state, training=False)  # No exploration
            next_state, reward, done, _ = env.step(action)
            next_state = env.state_to_index(next_state)
            
            episode_reward += reward
            episode_length += 1
            
            if done or episode_length > 100:
                break
            
            state = next_state
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
    
    print(f"Test Performance (100 episodes):")
    print(f"  Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Average Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
    print(f"  Success Rate: {np.mean([r > 5 for r in test_rewards]):.2%}")
    
    print("✅ Q-Learning implementation complete!")
    return agent, env


def exercise_2_sarsa_vs_qlearning():
    """
    Exercise 2: Compare SARSA vs Q-Learning
    
    Understand the difference between on-policy (SARSA) and 
    off-policy (Q-Learning) temporal difference methods.
    """
    print("\n⚖️ Exercise 2: SARSA vs Q-Learning Comparison")
    print("=" * 60)
    
    # Create environment with more challenging layout
    env = GridWorld(width=6, height=6, goal_reward=10.0, step_penalty=-0.1, obstacle_penalty=-2.0)
    
    # Add more obstacles to make the problem more interesting
    env.obstacles = {(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 1), (4, 2)}
    
    print("Environment: Challenging 6x6 Grid with obstacles")
    env.render()
    
    # Create both agents
    q_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    sarsa_agent = SARSAAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    # Train both agents
    n_episodes = 1000
    
    print(f"\nTraining both agents for {n_episodes} episodes...")
    
    # Train Q-Learning
    q_history = q_agent.train(env, n_episodes=n_episodes, verbose=False)
    
    # Train SARSA
    print("Training SARSA agent...")
    for episode in range(n_episodes):
        episode_reward, episode_length = sarsa_agent.train_episode(env)
        sarsa_agent.episode_rewards.append(episode_reward)
        sarsa_agent.episode_lengths.append(episode_length)
        
        if episode % 100 == 0:
            avg_reward = np.mean(sarsa_agent.episode_rewards[-100:])
            print(f"   Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    # Compare results
    plt.figure(figsize=(15, 10))
    
    # Learning curves comparison
    plt.subplot(2, 3, 1)
    window = 50
    q_rewards_smooth = np.convolve(q_agent.episode_rewards, np.ones(window)/window, mode='valid')
    sarsa_rewards_smooth = np.convolve(sarsa_agent.episode_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(q_rewards_smooth, label='Q-Learning', linewidth=2)
    plt.plot(sarsa_rewards_smooth, label='SARSA', linewidth=2)
    plt.title('Learning Curves (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Episode lengths comparison
    plt.subplot(2, 3, 2)
    q_lengths_smooth = np.convolve(q_agent.episode_lengths, np.ones(window)/window, mode='valid')
    sarsa_lengths_smooth = np.convolve(sarsa_agent.episode_lengths, np.ones(window)/window, mode='valid')
    
    plt.plot(q_lengths_smooth, label='Q-Learning', linewidth=2)
    plt.plot(sarsa_lengths_smooth, label='SARSA', linewidth=2)
    plt.title('Episode Lengths (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-table comparison
    plt.subplot(2, 3, 3)
    q_values_q = q_agent.q_table.reshape(env.height, env.width, env.n_actions)
    q_values_sarsa = sarsa_agent.q_table.reshape(env.height, env.width, env.n_actions)
    
    max_q_q = np.max(q_values_q, axis=2)
    max_q_sarsa = np.max(q_values_sarsa, axis=2)
    
    diff = max_q_q - max_q_sarsa
    im = plt.imshow(diff, cmap='RdBu', center=0)
    plt.colorbar(im)
    plt.title('Value Function Difference\n(Q-Learning - SARSA)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Policy comparison
    plt.subplot(2, 3, 4)
    policy_q = np.argmax(q_values_q, axis=2)
    policy_sarsa = np.argmax(q_values_sarsa, axis=2)
    
    # Show where policies differ
    policy_diff = (policy_q != policy_sarsa).astype(int)
    plt.imshow(policy_diff, cmap='Reds')
    plt.title('Policy Differences\n(Red = Different Actions)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Final performance comparison
    plt.subplot(2, 3, 5)
    final_q_rewards = q_agent.episode_rewards[-100:]
    final_sarsa_rewards = sarsa_agent.episode_rewards[-100:]
    
    plt.hist(final_q_rewards, bins=20, alpha=0.7, label='Q-Learning', edgecolor='black')
    plt.hist(final_sarsa_rewards, bins=20, alpha=0.7, label='SARSA', edgecolor='black')
    plt.title('Final Performance Distribution')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Risk analysis
    plt.subplot(2, 3, 6)
    
    def calculate_risk_metric(q_table, env):
        """Calculate how often agent chooses risky actions near obstacles"""
        risk_score = 0
        total_states = 0
        
        for y in range(env.height):
            for x in range(env.width):
                if (y, x) not in env.obstacles and (y, x) != env.goal_state:
                    state_idx = env.state_to_index((y, x))
                    best_action = np.argmax(q_table[state_idx])
                    
                    # Check if action leads toward obstacle
                    dy, dx = env.actions[best_action]
                    new_pos = (y + dy, x + dx)
                    
                    if new_pos in env.obstacles:
                        risk_score += 1
                    total_states += 1
        
        return risk_score / total_states if total_states > 0 else 0
    
    q_risk = calculate_risk_metric(q_agent.q_table, env)
    sarsa_risk = calculate_risk_metric(sarsa_agent.q_table, env)
    
    plt.bar(['Q-Learning', 'SARSA'], [q_risk, sarsa_risk], alpha=0.8)
    plt.title('Risk-Taking Behavior')
    plt.ylabel('Fraction of Risky Actions')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Q-Learning vs SARSA Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"Q-Learning:")
    print(f"  Final Average Reward: {np.mean(final_q_rewards):.2f} ± {np.std(final_q_rewards):.2f}")
    print(f"  Risk Score: {q_risk:.3f}")
    
    print(f"SARSA:")
    print(f"  Final Average Reward: {np.mean(final_sarsa_rewards):.2f} ± {np.std(final_sarsa_rewards):.2f}")
    print(f"  Risk Score: {sarsa_risk:.3f}")
    
    print("\n💡 Key Insights:")
    print("• Q-Learning (off-policy) learns the optimal policy regardless of exploration")
    print("• SARSA (on-policy) learns a safer policy that accounts for exploration")
    print("• SARSA typically shows more conservative behavior near obstacles")
    print("• Q-Learning may converge faster but can be more sensitive to exploration")
    
    print("✅ SARSA vs Q-Learning comparison complete!")
    return q_agent, sarsa_agent


def exercise_3_policy_gradients():
    """
    Exercise 3: Policy Gradient Methods (REINFORCE)
    
    Implement policy-based RL using the REINFORCE algorithm.
    Learn to optimize policies directly rather than value functions.
    """
    print("\n🎭 Exercise 3: Policy Gradient Methods (REINFORCE)")
    print("=" * 60)
    
    # Create simpler environment for policy gradients
    env = GridWorld(width=4, height=4, goal_reward=10.0, step_penalty=-0.1)
    env.obstacles = {(1, 1), (2, 2)}  # Simpler obstacle layout
    
    print("Environment: 4x4 Grid World for Policy Gradients")
    env.render()
    
    # Create policy gradient agent
    pg_agent = PolicyGradientAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.01,
        discount_factor=0.95
    )
    
    # For comparison, also train Q-learning agent
    q_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    print("\nTraining Policy Gradient Agent...")
    pg_history = pg_agent.train(env, n_episodes=1000, verbose=True)
    
    print("\nTraining Q-Learning Agent for comparison...")
    q_history = q_agent.train(env, n_episodes=1000, verbose=False)
    
    # Analyze policy evolution
    plt.figure(figsize=(15, 12))
    
    # Learning curves
    plt.subplot(3, 3, 1)
    plt.plot(pg_history['episodes'], pg_history['rewards'], label='REINFORCE', linewidth=2)
    plt.plot(q_history['episodes'], q_history['rewards'], label='Q-Learning', linewidth=2)
    plt.title('Learning Curves Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Policy visualization
    plt.subplot(3, 3, 2)
    
    # Extract policy from policy gradient agent
    pg_policy = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            state_idx = env.state_to_index((y, x))
            action_probs = pg_agent.softmax_policy(state_idx)
            pg_policy[y, x] = np.argmax(action_probs)
    
    plt.imshow(pg_policy, cmap='tab10')
    plt.title('REINFORCE Policy')
    
    # Add policy arrows
    action_symbols = ['↑', '↓', '←', '→']
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) not in env.obstacles and (y, x) != env.goal_state:
                action = int(pg_policy[y, x])
                plt.text(x, y, action_symbols[action], ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Q-Learning policy for comparison
    plt.subplot(3, 3, 3)
    q_values_reshaped = q_agent.q_table.reshape(env.height, env.width, env.n_actions)
    q_policy = np.argmax(q_values_reshaped, axis=2)
    
    plt.imshow(q_policy, cmap='tab10')
    plt.title('Q-Learning Policy')
    
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) not in env.obstacles and (y, x) != env.goal_state:
                action = q_policy[y, x]
                plt.text(x, y, action_symbols[action], ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Policy entropy analysis
    plt.subplot(3, 3, 4)
    
    def calculate_policy_entropy(agent, env):
        """Calculate entropy of policy at each state"""
        entropies = []
        for y in range(env.height):
            for x in range(env.width):
                if (y, x) not in env.obstacles:
                    state_idx = env.state_to_index((y, x))
                    action_probs = agent.softmax_policy(state_idx)
                    entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                    entropies.append(entropy)
        return np.mean(entropies)
    
    # Track entropy over time during training
    entropy_history = []
    temp_agent = PolicyGradientAgent(env.n_states, env.n_actions, 0.01, 0.95)
    
    for episode in range(0, 1000, 50):
        # Simulate training up to this episode
        temp_agent.policy_weights = pg_agent.policy_weights.copy()  # Approximate
        entropy = calculate_policy_entropy(temp_agent, env)
        entropy_history.append(entropy)
    
    plt.plot(range(0, 1000, 50), entropy_history, linewidth=2, color='purple')
    plt.title('Policy Entropy Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Entropy')
    plt.grid(True, alpha=0.3)
    
    # Action probability heatmaps for interesting states
    interesting_states = [(0, 0), (1, 0), (2, 1), (3, 3)]  # Various positions
    
    for i, (y, x) in enumerate(interesting_states):
        if (y, x) not in env.obstacles:
            plt.subplot(3, 3, 5 + i)
            state_idx = env.state_to_index((y, x))
            action_probs = pg_agent.softmax_policy(state_idx)
            
            plt.bar(range(env.n_actions), action_probs, alpha=0.8)
            plt.title(f'Action Probs at ({y},{x})')
            plt.xlabel('Action')
            plt.ylabel('Probability')
            plt.xticks(range(env.n_actions), action_symbols)
            plt.grid(True, alpha=0.3)
    
    # Variance analysis
    plt.subplot(3, 3, 9)
    
    # Calculate reward variance for both methods
    pg_variance = np.var(pg_agent.episode_rewards[-100:])
    q_variance = np.var(q_agent.episode_rewards[-100:])
    
    plt.bar(['REINFORCE', 'Q-Learning'], [pg_variance, q_variance], alpha=0.8)
    plt.title('Reward Variance (Final 100 Episodes)')
    plt.ylabel('Variance')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Policy Gradient Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Test stochastic vs deterministic policies
    print("\nTesting Policy Behavior:")
    
    def test_policy_stochasticity(agent, env, n_runs=10):
        """Test how stochastic the policy is"""
        trajectories = []
        
        for run in range(n_runs):
            state = env.reset()
            state = env.state_to_index(state)
            trajectory = [state]
            
            for _ in range(20):  # Max 20 steps
                if hasattr(agent, 'softmax_policy'):
                    action = agent.choose_action(state)
                else:
                    action = agent.choose_action(state, training=False)
                
                next_state, _, done, _ = env.step(action)
                next_state = env.state_to_index(next_state)
                trajectory.append(next_state)
                
                if done:
                    break
                state = next_state
            
            trajectories.append(trajectory)
        
        return trajectories
    
    pg_trajectories = test_policy_stochasticity(pg_agent, env)
    q_trajectories = test_policy_stochasticity(q_agent, env)
    
    print(f"REINFORCE trajectory diversity: {len(set(map(tuple, pg_trajectories)))}/{len(pg_trajectories)}")
    print(f"Q-Learning trajectory diversity: {len(set(map(tuple, q_trajectories)))}/{len(q_trajectories)}")
    
    print("\n💡 Key Insights:")
    print("• Policy gradients directly optimize the policy")
    print("• REINFORCE can learn stochastic policies")
    print("• Higher variance but can find globally optimal policies")
    print("• Suitable for continuous action spaces")
    
    print("✅ Policy gradient implementation complete!")
    return pg_agent, q_agent


def exercise_4_multi_armed_bandits():
    """
    Exercise 4: Multi-Armed Bandits and Exploration Strategies
    
    Explore the exploration vs exploitation trade-off using different
    bandit algorithms on the multi-armed bandit problem.
    """
    print("\n🎰 Exercise 4: Multi-Armed Bandits")
    print("=" * 60)
    
    # Create bandit environment
    n_arms = 10
    bandit = MultiArmedBandit(n_arms=n_arms, reward_std=1.0)
    
    print(f"Multi-Armed Bandit with {n_arms} arms")
    print(f"True arm means: {bandit.true_means}")
    print(f"Optimal arm: {bandit.optimal_arm} (mean: {bandit.true_means[bandit.optimal_arm]:.2f})")
    
    # Test different strategies
    strategies = {
        'ε-greedy (0.01)': {'strategy': 'epsilon_greedy', 'epsilon': 0.01},
        'ε-greedy (0.1)': {'strategy': 'epsilon_greedy', 'epsilon': 0.1},
        'ε-greedy (0.3)': {'strategy': 'epsilon_greedy', 'epsilon': 0.3},
        'UCB (c=1)': {'strategy': 'ucb', 'c': 1.0},
        'UCB (c=2)': {'strategy': 'ucb', 'c': 2.0},
        'Thompson Sampling': {'strategy': 'thompson'}
    }
    
    n_steps = 2000
    n_runs = 100  # Multiple runs for statistical significance
    
    print(f"\nRunning {n_runs} experiments with {n_steps} steps each...")
    
    results = {}
    
    for name, params in strategies.items():
        print(f"Testing {name}...")
        
        all_rewards = []
        all_regrets = []
        all_arm_selections = []
        
        for run in range(n_runs):
            # Reset bandit for each run
            bandit.reset()
            
            # Create agent
            agent = BanditAgent(n_arms=n_arms, **params)
            
            # Train agent
            history = agent.train(bandit, n_steps=n_steps)
            
            all_rewards.append(history['cumulative_rewards'])
            all_regrets.append(history['regrets'])
            all_arm_selections.append(history['arm_counts'])
        
        # Store results
        results[name] = {
            'rewards': np.array(all_rewards),
            'regrets': np.array(all_regrets),
            'arm_counts': np.array(all_arm_selections)
        }
    
    # Analyze results
    plt.figure(figsize=(15, 12))
    
    # Cumulative reward
    plt.subplot(2, 3, 1)
    for name, data in results.items():
        mean_rewards = np.mean(data['rewards'], axis=0)
        std_rewards = np.std(data['rewards'], axis=0)
        
        plt.plot(mean_rewards, label=name, linewidth=2)
        plt.fill_between(range(len(mean_rewards)), 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, alpha=0.3)
    
    plt.title('Cumulative Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative regret
    plt.subplot(2, 3, 2)
    for name, data in results.items():
        mean_regrets = np.mean(data['regrets'], axis=0)
        std_regrets = np.std(data['regrets'], axis=0)
        
        plt.plot(mean_regrets, label=name, linewidth=2)
        plt.fill_between(range(len(mean_regrets)), 
                        mean_regrets - std_regrets, 
                        mean_regrets + std_regrets, alpha=0.3)
    
    plt.title('Cumulative Regret')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Average reward over time
    plt.subplot(2, 3, 3)
    window = 100
    
    for name, data in results.items():
        rewards = data['rewards']
        # Convert cumulative to instantaneous
        instant_rewards = np.diff(rewards, axis=1, prepend=0)
        # Smooth with moving average
        smoothed = np.array([np.convolve(run, np.ones(window)/window, mode='valid') 
                            for run in instant_rewards])
        mean_smooth = np.mean(smoothed, axis=0)
        
        plt.plot(mean_smooth, label=name, linewidth=2)
    
    plt.title(f'Average Reward (Moving Window {window})')
    plt.xlabel('Time Step')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optimal arm selection frequency
    plt.subplot(2, 3, 4)
    
    optimal_frequencies = {}
    for name, data in results.items():
        arm_counts = data['arm_counts']
        total_pulls = np.sum(arm_counts, axis=2)
        optimal_pulls = arm_counts[:, :, bandit.optimal_arm]
        optimal_freq = optimal_pulls / total_pulls
        optimal_frequencies[name] = np.mean(optimal_freq, axis=0)
    
    for name, freq in optimal_frequencies.items():
        plt.plot(freq, label=name, linewidth=2)
    
    plt.title('Optimal Arm Selection Frequency')
    plt.xlabel('Run')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison
    plt.subplot(2, 3, 5)
    
    final_regrets = []
    strategy_names = []
    
    for name, data in results.items():
        final_regret = np.mean(data['regrets'][:, -1])
        final_regrets.append(final_regret)
        strategy_names.append(name)
    
    plt.bar(range(len(strategy_names)), final_regrets, alpha=0.8)
    plt.title('Final Cumulative Regret')
    plt.xlabel('Strategy')
    plt.ylabel('Final Regret')
    plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Arm distribution for best strategies
    plt.subplot(2, 3, 6)
    
    # Show arm selection distribution for best performing strategy
    best_strategy = strategy_names[np.argmin(final_regrets)]
    best_arm_counts = results[best_strategy]['arm_counts']
    mean_arm_counts = np.mean(best_arm_counts, axis=0)
    final_arm_dist = np.mean(mean_arm_counts, axis=0)
    
    colors = ['red' if i == bandit.optimal_arm else 'blue' for i in range(n_arms)]
    plt.bar(range(n_arms), final_arm_dist, color=colors, alpha=0.8)
    plt.title(f'Final Arm Distribution\n({best_strategy})')
    plt.xlabel('Arm')
    plt.ylabel('Average Pulls')
    plt.grid(True, alpha=0.3)
    
    # Add true means as reference
    ax2 = plt.gca().twinx()
    ax2.plot(range(n_arms), bandit.true_means, 'ko-', alpha=0.7, label='True Means')
    ax2.set_ylabel('True Mean Reward')