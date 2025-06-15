"""
Neural Odyssey - Week 32: Advanced Optimization for Foundation Models
Phase 2: Core Machine Learning (Week 20)

The Optimization Revolution: From Simple Gradients to Foundation Models

This week bridges traditional optimization theory with the cutting-edge techniques
that enable training trillion-parameter foundation models. You'll implement the
optimizers powering GPT, DALLE, and other modern AI systems, understanding both
the mathematical foundations and practical engineering challenges of large-scale
optimization.

Comprehensive exploration includes:
1. Advanced optimizers: AdamW, Lion, Sophia from mathematical foundations
2. Learning rate scheduling and warm-up strategies for stable training
3. Distributed optimization and gradient accumulation techniques
4. Memory-efficient training: gradient checkpointing and mixed precision
5. Foundation model-specific challenges and solutions
6. Multimodal optimization and cross-modal alignment
7. Optimization landscape analysis and loss surface visualization
8. Production optimization systems and monitoring

To get started, run: python exercises.py

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
torch.manual_seed(42)

print("🧠 Neural Odyssey - Week 32: Advanced Optimization for Foundation Models")
print("=" * 80)
print("Exploring the optimization techniques that power modern AI")
print("From mathematical foundations to trillion-parameter training")
print("=" * 80)


# ==========================================
# ADVANCED OPTIMIZERS FROM SCRATCH
# ==========================================

class AdamWOptimizer:
    """
    AdamW optimizer implementation from scratch with decoupled weight decay
    The optimization workhorse of foundation models
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for momentum and squared gradient averaging
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient (decoupled from gradients)
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize momentum and velocity
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0
    
    def step(self):
        """Perform single optimization step"""
        self.step_count += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay directly to parameters (decoupled)
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Update biased second moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            # Update parameters
            step_size = self.lr / bias_correction1
            denom = (self.v[i].sqrt() / np.sqrt(bias_correction2)).add_(self.eps)
            
            param.data.addcdiv_(self.m[i], denom, value=-step_size)
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class LionOptimizer:
    """
    Lion optimizer implementation - memory efficient alternative to AdamW
    Uses sign of gradient update for memory efficiency
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        """
        Initialize Lion optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate (typically smaller than AdamW)
            betas: Coefficients for momentum
            weight_decay: Weight decay coefficient
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
        # Initialize momentum (only momentum, no second moment)
        self.m = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        """Perform single optimization step"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Compute update direction using momentum
            update = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update parameters using sign of update
            param.data.add_(torch.sign(update), alpha=-self.lr)
            
            # Apply weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Update momentum
            self.m[i].mul_(self.beta2).add_(grad, alpha=1 - self.beta2)
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class SophiaOptimizer:
    """
    Sophia optimizer with Hessian diagonal approximation
    Second-order information for better convergence
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, rho=0.04, update_period=10):
        """
        Initialize Sophia optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for moment estimates
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient
            rho: Hessian regularization parameter
            update_period: Steps between Hessian diagonal updates
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rho = rho
        self.update_period = update_period
        
        # Initialize momentum and Hessian diagonal
        self.m = [torch.zeros_like(p) for p in self.params]
        self.h = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0
    
    def step(self, loss_fn=None):
        """
        Perform single optimization step
        
        Args:
            loss_fn: Loss function for Hessian computation
        """
        self.step_count += 1
        
        # Update Hessian diagonal periodically
        if self.step_count % self.update_period == 0 and loss_fn is not None:
            self._update_hessian_diagonal(loss_fn)
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Update momentum
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Bias correction for momentum
            bias_correction = 1 - self.beta1 ** self.step_count
            m_hat = self.m[i] / bias_correction
            
            # Update parameters using Hessian-adjusted gradients
            denom = torch.clamp(self.h[i], min=self.eps)
            param.data.add_(m_hat / (denom + self.rho), alpha=-self.lr)
    
    def _update_hessian_diagonal(self, loss_fn):
        """Update Hessian diagonal approximation"""
        # Compute gradients
        grads = torch.autograd.grad(loss_fn(), self.params, create_graph=True)
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Compute Hessian diagonal using second derivatives
            hessian_diag = torch.autograd.grad(
                grad.sum(), param, retain_graph=True, only_inputs=True
            )[0]
            
            # Update Hessian diagonal with momentum
            self.h[i].mul_(self.beta2).add_(
                hessian_diag.abs(), alpha=1 - self.beta2
            )
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ==========================================
# LEARNING RATE SCHEDULING
# ==========================================

class CosineWarmupScheduler:
    """
    Cosine annealing with warm-up - standard for foundation models
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        """
        Initialize cosine warm-up scheduler
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warm-up steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        return lr


class InverseSqrtScheduler:
    """
    Inverse square root decay - used in original Transformer
    """
    
    def __init__(self, optimizer, warmup_steps, factor=1.0):
        """
        Initialize inverse sqrt scheduler
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warm-up steps
            factor: Scaling factor
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Inverse square root decay
            lr = self.factor * self.base_lr * np.sqrt(self.warmup_steps) / np.sqrt(self.step_count)
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        return lr


# ==========================================
# FOUNDATION MODEL SIMULATOR
# ==========================================

class MiniTransformer(nn.Module):
    """
    Simplified transformer for optimization experiments
    Demonstrates foundation model optimization challenges
    """
    
    def __init__(self, vocab_size=1000, embed_dim=256, num_heads=8, 
                 num_layers=6, seq_length=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        """Forward pass"""
        seq_length = input_ids.size(1)
        
        # Create position indices
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        
        return logits


# ==========================================
# DISTRIBUTED TRAINING SIMULATION
# ==========================================

class GradientAccumulator:
    """
    Simulate gradient accumulation for large effective batch sizes
    Essential for memory-constrained foundation model training
    """
    
    def __init__(self, model, accumulation_steps=8):
        """
        Initialize gradient accumulator
        
        Args:
            model: Model to train
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = {}
        self.step_count = 0
    
    def accumulate_gradients(self, loss):
        """
        Accumulate gradients from loss
        
        Args:
            loss: Loss tensor (should be scaled by accumulation_steps)
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
    
    def should_update(self):
        """Check if gradients should be applied"""
        return self.step_count % self.accumulation_steps == 0
    
    def reset(self):
        """Reset accumulation counter"""
        self.step_count = 0


class MixedPrecisionTraining:
    """
    Simulate mixed precision training for memory efficiency
    """
    
    def __init__(self, model, optimizer, scaler=None):
        """
        Initialize mixed precision training
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scaler: Gradient scaler for fp16 training
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.use_amp = scaler is not None
    
    def forward_backward(self, inputs, targets, criterion):
        """
        Forward and backward pass with mixed precision
        
        Args:
            inputs: Input tensors
            targets: Target tensors
            criterion: Loss function
            
        Returns:
            Loss value
        """
        if self.use_amp and torch.cuda.is_available():
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Scale loss and backward
            self.scaler.scale(loss).backward()
            return loss.item()
        else:
            # Standard precision
            outputs = self.model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            return loss.item()
    
    def optimizer_step(self):
        """Perform optimizer step with gradient scaling"""
        if self.use_amp and torch.cuda.is_available():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()


# ==========================================
# OPTIMIZATION LANDSCAPE ANALYSIS
# ==========================================

class OptimizationLandscapeAnalyzer:
    """
    Analyze optimization landscapes and loss surfaces
    """
    
    def __init__(self, model, loss_fn):
        """
        Initialize landscape analyzer
        
        Args:
            model: Model to analyze
            loss_fn: Loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.parameter_history = []
        self.loss_history = []
        self.gradient_norms = []
    
    def log_step(self, loss_value):
        """Log optimization step"""
        # Save current parameters
        params = [p.clone().detach() for p in self.model.parameters()]
        self.parameter_history.append(params)
        self.loss_history.append(loss_value)
        
        # Compute gradient norm
        grad_norm = self._compute_gradient_norm()
        self.gradient_norms.append(grad_norm)
    
    def _compute_gradient_norm(self):
        """Compute gradient norm"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)
    
    def visualize_training_dynamics(self):
        """Visualize training dynamics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(self.loss_history)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norms
        axes[0, 1].plot(self.gradient_norms)
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss smoothing analysis
        if len(self.loss_history) > 10:
            window_size = max(10, len(self.loss_history) // 20)
            smoothed_loss = pd.Series(self.loss_history).rolling(window=window_size).mean()
            axes[1, 0].plot(self.loss_history, alpha=0.3, label='Raw')
            axes[1, 0].plot(smoothed_loss, label=f'Smoothed (window={window_size})')
            axes[1, 0].set_title('Loss Smoothing')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence analysis
        if len(self.loss_history) > 50:
            # Compute loss improvement rate
            improvement_rate = []
            window = 20
            for i in range(window, len(self.loss_history)):
                recent_loss = np.mean(self.loss_history[i-window:i])
                earlier_loss = np.mean(self.loss_history[i-2*window:i-window])
                if earlier_loss > 0:
                    improvement = (earlier_loss - recent_loss) / earlier_loss
                    improvement_rate.append(improvement)
            
            axes[1, 1].plot(improvement_rate)
            axes[1, 1].set_title('Loss Improvement Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Improvement Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_loss_surface(self, data_loader, num_points=50):
        """
        Analyze loss surface around current parameters
        
        Args:
            data_loader: Data loader for loss computation
            num_points: Number of points to sample
        """
        print("🔍 Analyzing loss surface...")
        
        # Get current parameters
        current_params = [p.clone() for p in self.model.parameters()]
        
        # Generate random directions
        directions = []
        for p in current_params:
            direction = torch.randn_like(p)
            direction = direction / direction.norm()
            directions.append(direction)
        
        # Sample points along random directions
        alphas = np.linspace(-2.0, 2.0, num_points)
        losses = []
        
        for alpha in alphas:
            # Move parameters
            for p, direction in zip(self.model.parameters(), directions):
                p.data = current_params[len(losses) % len(current_params)] + alpha * direction
            
            # Compute loss
            total_loss = 0
            count = 0
            with torch.no_grad():
                for inputs, targets in data_loader:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    total_loss += loss.item()
                    count += 1
                    if count >= 5:  # Sample a few batches
                        break
            
            losses.append(total_loss / count)
        
        # Restore original parameters
        for p, orig_p in zip(self.model.parameters(), current_params):
            p.data = orig_p
        
        # Plot loss surface
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, losses, 'b-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Current position')
        plt.xlabel('Step size')
        plt.ylabel('Loss')
        plt.title('Loss Surface Along Random Direction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return alphas, losses


# ==========================================
# COMPREHENSIVE OPTIMIZATION EXPERIMENTS
# ==========================================

class FoundationModelOptimizationExperiments:
    """
    Comprehensive experiments comparing foundation model optimization techniques
    """
    
    def __init__(self):
        """Initialize experiment framework"""
        self.results = {}
        self.models = {}
        
        # Create synthetic dataset for transformer training
        self.vocab_size = 1000
        self.seq_length = 64
        self.batch_size = 32
        
        # Generate synthetic language modeling data
        self.train_data, self.val_data = self._create_synthetic_language_data()
        
        print("🏗️  Foundation Model Optimization Experiments Initialized")
        print(f"   Dataset: {len(self.train_data)} training sequences")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Sequence length: {self.seq_length}")
    
    def _create_synthetic_language_data(self, num_sequences=10000):
        """Create synthetic language modeling dataset"""
        # Generate sequences with some structure
        sequences = []
        for _ in range(num_sequences):
            # Create structured sequences (simple patterns)
            seq = []
            for i in range(self.seq_length):
                if i < 10:
                    # Start with special tokens
                    token = np.random.randint(0, 50)
                elif i < 30:
                    # Middle with more diversity
                    token = np.random.randint(0, self.vocab_size)
                else:
                    # End with patterns
                    token = (i * 37) % 100 + np.random.randint(0, 10)
                seq.append(token)
            sequences.append(seq)
        
        sequences = torch.tensor(sequences, dtype=torch.long)
        
        # Split into train/validation
        split_idx = int(0.8 * len(sequences))
        train_data = sequences[:split_idx]
        val_data = sequences[split_idx:]
        
        return train_data, val_data
    
    def experiment_optimizer_comparison(self):
        """Compare different optimizers on foundation model training"""
        print("\n🔬 Experiment 1: Optimizer Comparison")
        print("=" * 50)
        
        optimizers = {
            'AdamW': lambda model: AdamWOptimizer(model.parameters(), lr=1e-3),
            'Lion': lambda model: LionOptimizer(model.parameters(), lr=1e-4),
            'Sophia': lambda model: SophiaOptimizer(model.parameters(), lr=1e-3)
        }
        
        results = {}
        
        for opt_name, opt_factory in optimizers.items():
            print(f"\n📊 Training with {opt_name}...")
            
            # Create model
            model = MiniTransformer(
                vocab_size=self.vocab_size,
                embed_dim=128,
                num_heads=4,
                num_layers=3,
                seq_length=self.seq_length
            )
            
            # Create optimizer
            optimizer = opt_factory(model)
            
            # Create scheduler
            scheduler = CosineWarmupScheduler(
                optimizer, 
                warmup_steps=100, 
                total_steps=1000
            )
            
            # Training loop
            model.train()
            criterion = nn.CrossEntropyLoss()
            losses = []
            learning_rates = []
            
            train_loader = DataLoader(
                TensorDataset(self.train_data[:-1], self.train_data[1:]),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            for epoch in range(3):  # Short training for comparison
                epoch_losses = []
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if batch_idx >= 100:  # Limit batches for quick comparison
                        break
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    losses.append(loss.item())
                    learning_rates.append(scheduler.optimizer.lr if hasattr(scheduler.optimizer, 'lr') 
                                        else scheduler.optimizer.param_groups[0]['lr'])
                    epoch_losses.append(loss.item())
                
                print(f"   Epoch {epoch + 1}: Loss = {np.mean(epoch_losses):.4f}")
            
            results[opt_name] = {
                'losses': losses,
                'learning_rates': learning_rates,
                'final_loss': losses[-1] if losses else float('inf'),
                'model': model
            }
        
        # Visualize comparison
        self._plot_optimizer_comparison(results)
        
        self.results['optimizer_comparison'] = results
        return results
    
    def experiment_learning_rate_scheduling(self):
        """Compare different learning rate scheduling strategies"""
        print("\n🔬 Experiment 2: Learning Rate Scheduling")
        print("=" * 50)
        
        schedulers = {
            'Cosine Warmup': lambda opt: CosineWarmupScheduler(opt, 100, 1000),
            'Inverse Sqrt': lambda opt: InverseSqrtScheduler(opt, 100),
            'Constant': lambda opt: None  # No scheduling
        }
        
        results = {}
        
        for sched_name, sched_factory in schedulers.items():
            print(f"\n📊 Training with {sched_name} scheduling...")
            
            # Create model and optimizer
            model = MiniTransformer(
                vocab_size=self.vocab_size,
                embed_dim=128,
                num_heads=4,
                num_layers=3,
                seq_length=self.seq_length
            )
            
            optimizer = AdamWOptimizer(model.parameters(), lr=1e-3)
            scheduler = sched_factory(optimizer) if sched_factory else None
            
            # Training loop
            model.train()
            criterion = nn.CrossEntropyLoss()
            losses = []
            learning_rates = []         
            
            train_loader = DataLoader(
                TensorDataset(self.train_data[:-1], self.train_data[1:]),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            for epoch in range(3):
                epoch_losses = []
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if batch_idx >= 100:
                        break
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    if scheduler:
                        scheduler.step()
                    
                    losses.append(loss.item())
                    current_lr = (scheduler.optimizer.lr if scheduler and hasattr(scheduler.optimizer, 'lr') 
                                else optimizer.lr if hasattr(optimizer, 'lr') 
                                else optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups')
                                else 1e-3)
                    learning_rates.append(current_lr)
                    epoch_losses.append(loss.item())
                
                print(f"   Epoch {epoch + 1}: Loss = {np.mean(epoch_losses):.4f}")
            
            results[sched_name] = {
                'losses': losses,
                'learning_rates': learning_rates,
                'final_loss': losses[-1] if losses else float('inf')
            }
        
        # Visualize scheduling comparison
        self._plot_scheduling_comparison(results)
        
        self.results['scheduling_comparison'] = results
        return results
    
    def experiment_gradient_accumulation(self):
        """Demonstrate gradient accumulation for large effective batch sizes"""
        print("\n🔬 Experiment 3: Gradient Accumulation")
        print("=" * 50)
        
        accumulation_steps = [1, 2, 4, 8]
        results = {}
        
        for acc_steps in accumulation_steps:
            print(f"\n📊 Training with {acc_steps} accumulation steps...")
            
            # Create model
            model = MiniTransformer(
                vocab_size=self.vocab_size,
                embed_dim=128,
                num_heads=4,
                num_layers=3,
                seq_length=self.seq_length
            )
            
            optimizer = AdamWOptimizer(model.parameters(), lr=1e-3)
            accumulator = GradientAccumulator(model, acc_steps)
            
            # Training with gradient accumulation
            model.train()
            criterion = nn.CrossEntropyLoss()
            losses = []
            
            # Smaller batch size to simulate memory constraints
            small_batch_size = self.batch_size // acc_steps
            train_loader = DataLoader(
                TensorDataset(self.train_data[:-1], self.train_data[1:]),
                batch_size=small_batch_size,
                shuffle=True
            )
            
            step_count = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 200:  # Limit for comparison
                    break
                
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Accumulate gradients
                accumulator.accumulate_gradients(loss)
                
                # Update when accumulation is complete
                if accumulator.should_update():
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1
                    
                    # Log effective batch loss
                    effective_loss = loss.item() * acc_steps  # Approximate
                    losses.append(effective_loss)
                    
                    if step_count % 10 == 0:
                        print(f"   Step {step_count}: Loss = {effective_loss:.4f}")
            
            results[f"acc_{acc_steps}"] = {
                'losses': losses,
                'accumulation_steps': acc_steps,
                'effective_batch_size': small_batch_size * acc_steps,
                'final_loss': losses[-1] if losses else float('inf')
            }
        
        # Visualize accumulation comparison
        self._plot_accumulation_comparison(results)
        
        self.results['accumulation_comparison'] = results
        return results
    
    def experiment_mixed_precision_training(self):
        """Demonstrate mixed precision training benefits"""
        print("\n🔬 Experiment 4: Mixed Precision Training")
        print("=" * 50)
        
        precision_modes = ['fp32', 'mixed_precision']
        results = {}
        
        for mode in precision_modes:
            print(f"\n📊 Training with {mode}...")
            
            # Create model
            model = MiniTransformer(
                vocab_size=self.vocab_size,
                embed_dim=128,
                num_heads=4,
                num_layers=3,
                seq_length=self.seq_length
            )
            
            optimizer = AdamWOptimizer(model.parameters(), lr=1e-3)
            
            # Setup mixed precision training
            use_amp = mode == 'mixed_precision'
            scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
            mp_trainer = MixedPrecisionTraining(model, optimizer, scaler)
            
            # Training
            model.train()
            criterion = nn.CrossEntropyLoss()
            losses = []
            training_times = []
            
            train_loader = DataLoader(
                TensorDataset(self.train_data[:-1], self.train_data[1:]),
                batch_size=self.batch_size,
                shuffle=True
            )
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 100:
                    break
                
                start_time = time.time()
                
                optimizer.zero_grad()
                loss = mp_trainer.forward_backward(inputs, targets, criterion)
                mp_trainer.optimizer_step()
                
                end_time = time.time()
                
                losses.append(loss)
                training_times.append(end_time - start_time)
                
                if batch_idx % 20 == 0:
                    print(f"   Batch {batch_idx}: Loss = {loss:.4f}, Time = {end_time - start_time:.4f}s")
            
            results[mode] = {
                'losses': losses,
                'training_times': training_times,
                'avg_time_per_batch': np.mean(training_times),
                'final_loss': losses[-1] if losses else float('inf')
            }
        
        # Visualize mixed precision comparison
        self._plot_mixed_precision_comparison(results)
        
        self.results['mixed_precision_comparison'] = results
        return results
    
    def experiment_optimization_landscape_analysis(self):
        """Analyze optimization landscape and training dynamics"""
        print("\n🔬 Experiment 5: Optimization Landscape Analysis")
        print("=" * 50)
        
        # Create model for analysis
        model = MiniTransformer(
            vocab_size=self.vocab_size,
            embed_dim=128,
            num_heads=4,
            num_layers=3,
            seq_length=self.seq_length
        )
        
        optimizer = AdamWOptimizer(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Create landscape analyzer
        analyzer = OptimizationLandscapeAnalyzer(model, criterion)
        
        # Training with landscape tracking
        model.train()
        train_loader = DataLoader(
            TensorDataset(self.train_data[:-1], self.train_data[1:]),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        print("🏃 Training model and analyzing landscape...")
        for epoch in range(2):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 150:
                    break
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                loss.backward()
                optimizer.step()
                
                # Log step for analysis
                analyzer.log_step(loss.item())
                
                if batch_idx % 30 == 0:
                    print(f"   Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # Visualize training dynamics
        analyzer.visualize_training_dynamics()
        
        # Analyze loss surface
        val_loader = DataLoader(
            TensorDataset(self.val_data[:-1], self.val_data[1:]),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        alphas, losses = analyzer.analyze_loss_surface(val_loader)
        
        self.results['landscape_analysis'] = {
            'analyzer': analyzer,
            'loss_surface': (alphas, losses)
        }
        
        return analyzer
    
    def _plot_optimizer_comparison(self, results):
        """Plot optimizer comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax = axes[0, 0]
        for opt_name, result in results.items():
            ax.plot(result['losses'], label=opt_name, alpha=0.8)
        ax.set_title('Training Loss Comparison')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rates
        ax = axes[0, 1]
        for opt_name, result in results.items():
            if 'learning_rates' in result:
                ax.plot(result['learning_rates'], label=opt_name, alpha=0.8)
        ax.set_title('Learning Rate Evolution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final loss comparison
        ax = axes[1, 0]
        opt_names = list(results.keys())
        final_losses = [results[name]['final_loss'] for name in opt_names]
        bars = ax.bar(opt_names, final_losses, alpha=0.7)
        ax.set_title('Final Loss Comparison')
        ax.set_ylabel('Final Loss')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.4f}', ha='center', va='bottom')
        
        # Convergence rate analysis
        ax = axes[1, 1]
        for opt_name, result in results.items():
            losses = result['losses']
            if len(losses) > 20:
                # Compute moving average for smoother convergence analysis
                window = 10
                smoothed = pd.Series(losses).rolling(window=window).mean()
                ax.plot(smoothed, label=f'{opt_name} (smoothed)', alpha=0.8)
        ax.set_title('Convergence Analysis (Smoothed)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_scheduling_comparison(self, results):
        """Plot learning rate scheduling comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Learning rate schedules
        ax = axes[0, 0]
        for sched_name, result in results.items():
            ax.plot(result['learning_rates'], label=sched_name, alpha=0.8)
        ax.set_title('Learning Rate Schedules')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss curves
        ax = axes[0, 1]
        for sched_name, result in results.items():
            ax.plot(result['losses'], label=sched_name, alpha=0.8)
        ax.set_title('Training Loss with Different Schedules')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final loss comparison
        ax = axes[1, 0]
        sched_names = list(results.keys())
        final_losses = [results[name]['final_loss'] for name in sched_names]
        bars = ax.bar(sched_names, final_losses, alpha=0.7)
        ax.set_title('Final Loss by Schedule')
        ax.set_ylabel('Final Loss')
        ax.grid(True, alpha=0.3)
        
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.4f}', ha='center', va='bottom')
        
        # Learning rate vs loss correlation
        ax = axes[1, 1]
        for sched_name, result in results.items():
            if len(result['learning_rates']) == len(result['losses']):
                ax.scatter(result['learning_rates'], result['losses'], 
                          label=sched_name, alpha=0.6, s=10)
        ax.set_title('Learning Rate vs Loss')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_accumulation_comparison(self, results):
        """Plot gradient accumulation comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves by accumulation steps
        ax = axes[0, 0]
        for name, result in results.items():
            acc_steps = result['accumulation_steps']
            ax.plot(result['losses'], label=f'{acc_steps} steps', alpha=0.8)
        ax.set_title('Loss by Accumulation Steps')
        ax.set_xlabel('Effective Update Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Effective batch size vs final loss
        ax = axes[0, 1]
        acc_steps = [results[name]['accumulation_steps'] for name in results.keys()]
        effective_batch_sizes = [results[name]['effective_batch_size'] for name in results.keys()]
        final_losses = [results[name]['final_loss'] for name in results.keys()]
        
        ax.scatter(effective_batch_sizes, final_losses, s=100, alpha=0.7)
        for i, steps in enumerate(acc_steps):
            ax.annotate(f'{steps} steps', 
                       (effective_batch_sizes[i], final_losses[i]),
                       xytext=(5, 5), textcoords='offset points')
        ax.set_title('Effective Batch Size vs Final Loss')
        ax.set_xlabel('Effective Batch Size')
        ax.set_ylabel('Final Loss')
        ax.grid(True, alpha=0.3)
        
        # Memory efficiency simulation
        ax = axes[1, 0]
        memory_usage = [self.batch_size / result['accumulation_steps'] for result in results.values()]
        throughput = [result['accumulation_steps'] for result in results.values()]
        
        bars = ax.bar(range(len(memory_usage)), memory_usage, alpha=0.7)
        ax.set_title('Simulated Memory Usage')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Memory per Microbatch')
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([f"{list(results.keys())[i]}" for i in range(len(results))])
        ax.grid(True, alpha=0.3)
        
        # Convergence comparison
        ax = axes[1, 1]
        for name, result in results.items():
            losses = result['losses']
            if len(losses) > 10:
                # Normalize by length for fair comparison
                steps = np.linspace(0, 1, len(losses))
                ax.plot(steps, losses, label=f"{result['accumulation_steps']} steps", alpha=0.8)
        ax.set_title('Normalized Convergence')
        ax.set_xlabel('Normalized Training Progress')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_mixed_precision_comparison(self, results):
        """Plot mixed precision training comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss comparison
        ax = axes[0, 0]
        for mode, result in results.items():
            ax.plot(result['losses'], label=mode, alpha=0.8)
        ax.set_title('Loss: FP32 vs Mixed Precision')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training time comparison
        ax = axes[0, 1]
        modes = list(results.keys())
        avg_times = [results[mode]['avg_time_per_batch'] for mode in modes]
        bars = ax.bar(modes, avg_times, alpha=0.7)
        ax.set_title('Average Time per Batch')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.4f}s', ha='center', va='bottom')
        
        # Speedup calculation
        if 'fp32' in results and 'mixed_precision' in results:
            speedup = results['fp32']['avg_time_per_batch'] / results['mixed_precision']['avg_time_per_batch']
            ax.text(0.5, max(avg_times) * 0.8, f'Speedup: {speedup:.2f}x', 
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Training time distribution
        ax = axes[1, 0]
        for mode, result in results.items():
            ax.hist(result['training_times'], alpha=0.6, label=mode, bins=20)
        ax.set_title('Training Time Distribution')
        ax.set_xlabel('Time per Batch (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Memory efficiency (simulated)
        ax = axes[1, 1]
        # Simulate memory usage (mixed precision typically uses ~50% less memory)
        memory_usage = {
            'fp32': 100,
            'mixed_precision': 60
        }
        
        modes = list(memory_usage.keys())
        memory_vals = list(memory_usage.values())
        bars = ax.bar(modes, memory_vals, alpha=0.7, color=['red', 'green'])
        ax.set_title('Simulated Memory Usage')
        ax.set_ylabel('Relative Memory Usage (%)')
        ax.grid(True, alpha=0.3)
        
        for bar, mem_val in zip(bars, memory_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem_val}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


# ==========================================
# MAIN EXECUTION AND DEMONSTRATIONS
# ==========================================

def demonstrate_advanced_optimizers():
    """Demonstrate advanced optimizer implementations"""
    print("\n🚀 Advanced Optimizers Demonstration")
    print("=" * 50)
    
    # Create simple test problem
    X = torch.randn(1000, 10)
    true_weights = torch.randn(10)
    y = X @ true_weights + 0.1 * torch.randn(1000)
    
    # Simple linear model
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()
    
    optimizers = {
        'AdamW': AdamWOptimizer(model.parameters(), lr=1e-2),
        'Lion': LionOptimizer(model.parameters(), lr=1e-3),
        'Sophia': SophiaOptimizer(model.parameters(), lr=1e-2)
    }
    
    print("🔍 Comparing optimizers on linear regression problem...")
    
    results = {}
    for opt_name, optimizer in optimizers.items():
        print(f"\n   Training with {opt_name}...")
        
        # Reset model
        model = nn.Linear(10, 1)
        optimizer = optimizers[opt_name].__class__(
            model.parameters(), 
            **optimizers[opt_name].__dict__
        )
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X).squeeze()
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            
            # Special handling for Sophia (needs loss function)
            if opt_name == 'Sophia' and hasattr(optimizer, 'step'):
                loss_fn = lambda: criterion(model(X).squeeze(), y)
                optimizer.step(loss_fn)
            else:
                optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 25 == 0:
                print(f"     Epoch {epoch}: Loss = {loss.item():.6f}")
        
        results[opt_name] = losses
        print(f"   Final loss: {losses[-1]:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for opt_name, losses in results.items():
        plt.plot(losses, label=opt_name, alpha=0.8)
    plt.title('Optimizer Comparison: Linear Regression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    final_losses = [losses[-1] for losses in results.values()]
    bars = plt.bar(results.keys(), final_losses, alpha=0.7)
    plt.title('Final Loss Comparison')
    plt.ylabel('Final Loss')
    plt.grid(True, alpha=0.3)
    
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Advanced optimizers demonstration completed!")


def demonstrate_learning_rate_scheduling():
    """Demonstrate learning rate scheduling strategies"""
    print("\n📈 Learning Rate Scheduling Demonstration")
    print("=" * 50)
    
    # Create dummy optimizer for demonstration
    class DummyOptimizer:
        def __init__(self, lr=1e-3):
            self.lr = lr
    
    optimizer = DummyOptimizer(lr=1e-3)
    
    schedulers = {
        'Cosine Warmup': CosineWarmupScheduler(optimizer, warmup_steps=50, total_steps=500),
        'Inverse Sqrt': InverseSqrtScheduler(optimizer, warmup_steps=50)
    }
    
    # Simulate training steps
    steps = 500
    results = {}
    
    for sched_name, scheduler in schedulers.items():
        optimizer.lr = 1e-3  # Reset
        learning_rates = []
        
        for step in range(steps):
            lr = scheduler.step()
            learning_rates.append(lr)
        
        results[sched_name] = learning_rates
    
    # Add constant learning rate for comparison
    results['Constant'] = [1e-3] * steps
    
    # Plot schedules
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for sched_name, lrs in results.items():
        plt.plot(lrs, label=sched_name, alpha=0.8, linewidth=2)
    plt.title('Learning Rate Schedules Comparison')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for sched_name, lrs in results.items():
        if sched_name != 'Constant':  # Skip constant for log scale
            plt.plot(lrs, label=sched_name, alpha=0.8, linewidth=2)
    plt.title('Learning Rate Schedules (Log Scale)')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Learning rate scheduling demonstration completed!")


def main():
    """Main execution function"""
    print("🎯 Starting Advanced Optimization Experiments...")
    
    # Initialize experiment framework
    experiments = FoundationModelOptimizationExperiments()
    
    print("\n🔬 Running Comprehensive Optimization Experiments")
    print("=" * 60)
    
    try:
        # Run all experiments
        print("\n1️⃣ Optimizer Comparison Experiment")
        experiments.experiment_optimizer_comparison()
        
        print("\n2️⃣ Learning Rate Scheduling Experiment")
        experiments.experiment_learning_rate_scheduling()
        
        print("\n3️⃣ Gradient Accumulation Experiment")
        experiments.experiment_gradient_accumulation()
        
        print("\n4️⃣ Mixed Precision Training Experiment")
        experiments.experiment_mixed_precision_training()
        
        print("\n5️⃣ Optimization Landscape Analysis")
        experiments.experiment_optimization_landscape_analysis()
        
        # Additional demonstrations
        print("\n🚀 Additional Demonstrations")
        demonstrate_advanced_optimizers()
        demonstrate_learning_rate_scheduling()
        
        print("\n" + "=" * 80)
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of Key Insights:")
        print("• AdamW remains the gold standard for foundation model training")
        print("• Lion offers memory efficiency with competitive performance")
        print("• Sophia shows promise with second-order information")
        print("• Learning rate warm-up is crucial for stable training")
        print("• Gradient accumulation enables large effective batch sizes")
        print("• Mixed precision training provides significant speedups")
        print("• Optimization landscape analysis reveals training dynamics")
        print("\n🔬 Ready for Phase 3: Advanced Deep Learning!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("Check your implementation and try again!")


if __name__ == "__main__":
    main()