"""
Neural Odyssey - Week 2: Calculus Interactive Visualizations
Browser-executable Python code for understanding optimization and gradient-based learning

This module provides interactive visualizations for understanding:
- Gradient descent optimization in 1D and 2D landscapes
- Learning rate effects on convergence and stability
- Momentum and adaptive optimization methods
- Chain rule and backpropagation in neural networks
- Real-time optimization dynamics and convergence analysis

Designed for browser execution via Pyodide with full interactivity.
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import math
from typing import Tuple, List, Optional, Callable

# Configure matplotlib for better browser display
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#2d2d2d'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

class CalculusVisualizer:
    """Main class for calculus and optimization visualizations"""
    
    def __init__(self):
        self.current_demo = None
        self.animation = None
        self.optimization_history = []
        
    def gradient_descent_explorer(self):
        """
        Interactive gradient descent landscape explorer
        Shows optimization paths on different functions with tunable parameters
        """
        print("🎨 Starting Gradient Descent Landscape Explorer...")
        
        # Create figure with complex layout
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
        
        # Main optimization plot
        ax_main = fig.add_subplot(gs[0, :2])
        ax_3d = fig.add_subplot(gs[0, 2], projection='3d')
        
        # Control panels
        ax_controls = fig.add_subplot(gs[1, :])
        ax_info = fig.add_subplot(gs[2, :])
        
        fig.suptitle('🚀 Gradient Descent Optimization Explorer', fontsize=16, color='cyan')
        
        # Define test functions with their gradients
        self.functions = {
            'Simple Quadratic': {
                'func': lambda x: (x - 2)**2 + 1,
                'grad': lambda x: 2*(x - 2),
                'domain': (-1, 5),
                'minimum': 2.0,
                'difficulty': 'Easy'
            },
            'Steep Valley': {
                'func': lambda x: 0.1*(x - 2)**2 + 10*np.sin(x)**2,
                'grad': lambda x: 0.2*(x - 2) + 20*np.sin(x)*np.cos(x),
                'domain': (-2, 6),
                'minimum': 2.0,
                'difficulty': 'Medium'
            },
            'Multiple Minima': {
                'func': lambda x: x**4 - 4*x**3 + 4*x**2 + 1,
                'grad': lambda x: 4*x**3 - 12*x**2 + 8*x,
                'domain': (-1, 4),
                'minimum': 'Multiple',
                'difficulty': 'Hard'
            },
            'Noisy Landscape': {
                'func': lambda x: (x - 2)**2 + 0.5*np.sin(10*x) + 1,
                'grad': lambda x: 2*(x - 2) + 5*np.cos(10*x),
                'domain': (0, 4),
                'minimum': '~2.0',
                'difficulty': 'Very Hard'
            }
        }
        
        # Initial parameters
        current_function = 'Simple Quadratic'
        learning_rate = 0.1
        momentum = 0.0
        start_point = -0.5
        
        # Plot initial function
        func_data = self.functions[current_function]
        x_range = np.linspace(func_data['domain'][0], func_data['domain'][1], 1000)
        y_values = func_data['func'](x_range)
        
        ax_main.clear()
        ax_main.plot(x_range, y_values, 'w-', linewidth=2, label=f'{current_function}')
        ax_main.set_xlabel('x')
        ax_main.set_ylabel('f(x)')
        ax_main.set_title('Optimization Landscape', color='lightblue')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Initialize optimization tracking
        optimization_path = []
        velocity = 0.0  # For momentum
        
        # Create sliders
        plt.subplots_adjust(bottom=0.35)
        
        # Slider positions
        slider_lr = Slider(plt.axes([0.1, 0.25, 0.3, 0.03]), 
                          'Learning Rate', 0.001, 0.5, valinit=learning_rate, valfmt='%.3f')
        slider_momentum = Slider(plt.axes([0.1, 0.21, 0.3, 0.03]),
                                'Momentum', 0.0, 0.99, valinit=momentum, valfmt='%.2f')
        slider_start = Slider(plt.axes([0.1, 0.17, 0.3, 0.03]),
                             'Start Point', func_data['domain'][0], func_data['domain'][1], 
                             valinit=start_point, valfmt='%.2f')
        
        # Function selection radio buttons
        ax_radio = plt.axes([0.5, 0.15, 0.2, 0.15])
        radio_functions = RadioButtons(ax_radio, list(self.functions.keys()))
        radio_functions.set_active(0)
        
        # Control buttons
        btn_run = Button(plt.axes([0.75, 0.23, 0.1, 0.04]), 'Run Optimization', color='green')
        btn_reset = Button(plt.axes([0.75, 0.18, 0.1, 0.04]), 'Reset', color='red')
        btn_step = Button(plt.axes([0.75, 0.13, 0.1, 0.04]), 'Single Step', color='blue')
        
        # Info display
        info_text = ax_info.text(0.02, 0.5, '', transform=ax_info.transAxes, fontsize=10,
                                verticalalignment='center', 
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        
        def update_function(label):
            """Update the displayed function"""
            nonlocal current_function, func_data, x_range, y_values
            current_function = label
            func_data = self.functions[current_function]
            x_range = np.linspace(func_data['domain'][0], func_data['domain'][1], 1000)
            y_values = func_data['func'](x_range)
            
            # Update slider ranges
            slider_start.valmin = func_data['domain'][0]
            slider_start.valmax = func_data['domain'][1]
            slider_start.set_val((func_data['domain'][0] + func_data['domain'][1]) / 2)
            
            redraw_function()
        
        def redraw_function():
            """Redraw the function and optimization path"""
            ax_main.clear()
            ax_main.plot(x_range, y_values, 'w-', linewidth=2, label=current_function)
            
            # Plot optimization path if it exists
            if optimization_path:
                path_x = [point[0] for point in optimization_path]
                path_y = [point[1] for point in optimization_path]
                
                # Plot path
                ax_main.plot(path_x, path_y, 'ro-', alpha=0.7, linewidth=2, markersize=4, 
                           label=f'Optimization Path ({len(path_x)} steps)')
                
                # Highlight start and end
                ax_main.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
                if len(path_x) > 1:
                    ax_main.plot(path_x[-1], path_y[-1], 'bo', markersize=10, label='Current')
                
                # Plot gradient arrows
                if len(path_x) > 1:
                    for i in range(min(len(path_x)-1, 10)):  # Show last 10 gradient arrows
                        x_pos = path_x[i]
                        grad = func_data['grad'](x_pos)
                        arrow_scale = 0.1
                        ax_main.arrow(x_pos, path_y[i], -arrow_scale * grad, 0,
                                    head_width=0.1, head_length=0.05, fc='yellow', ec='yellow', alpha=0.7)
            
            ax_main.set_xlabel('x')
            ax_main.set_ylabel('f(x)')
            ax_main.set_title(f'Optimization Landscape: {current_function}', color='lightblue')
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
            
            # Update info display
            if optimization_path:
                current_x = optimization_path[-1][0]
                current_y = optimization_path[-1][1]
                current_grad = func_data['grad'](current_x)
                
                info = f'''📊 Optimization Status:
Current Position: x = {current_x:.4f}, f(x) = {current_y:.4f}
Gradient: f'(x) = {current_grad:.4f}
Steps Taken: {len(optimization_path)}
Learning Rate: {slider_lr.val:.3f}
Momentum: {slider_momentum.val:.2f}

🎯 Function Info:
Name: {current_function}
Difficulty: {func_data['difficulty']}
True Minimum: {func_data['minimum']}
Domain: {func_data['domain']}

💡 Tips:
• Large gradients → steep slopes
• Small learning rates → slow but stable
• High momentum → faster convergence'''
                info_text.set_text(info)
            else:
                info_text.set_text(f'🚀 Ready to optimize {current_function}!\nClick "Run Optimization" to start.')
            
            plt.draw()
        
        def run_optimization(event):
            """Run complete optimization"""
            nonlocal optimization_path, velocity
            
            # Reset optimization
            optimization_path = []
            velocity = 0.0
            
            # Get current parameters
            lr = slider_lr.val
            mom = slider_momentum.val
            x_current = slider_start.val
            
            # Run optimization steps
            max_steps = 100
            tolerance = 1e-6
            
            for step in range(max_steps):
                y_current = func_data['func'](x_current)
                grad_current = func_data['grad'](x_current)
                
                optimization_path.append((x_current, y_current))
                
                # Check convergence
                if abs(grad_current) < tolerance:
                    print(f"✅ Converged in {step+1} steps!")
                    break
                
                # Gradient descent with momentum
                velocity = mom * velocity - lr * grad_current
                x_current = x_current + velocity
                
                # Keep within domain bounds
                x_current = np.clip(x_current, func_data['domain'][0], func_data['domain'][1])
            
            redraw_function()
        
        def reset_optimization(event):
            """Reset optimization path"""
            nonlocal optimization_path, velocity
            optimization_path = []
            velocity = 0.0
            redraw_function()
        
        def single_step(event):
            """Take a single optimization step"""
            nonlocal optimization_path, velocity
            
            if not optimization_path:
                # Initialize
                velocity = 0.0
                x_current = slider_start.val
            else:
                x_current = optimization_path[-1][0]
            
            # Get current parameters
            lr = slider_lr.val
            mom = slider_momentum.val
            
            # Calculate gradient and step
            y_current = func_data['func'](x_current)
            grad_current = func_data['grad'](x_current)
            
            optimization_path.append((x_current, y_current))
            
            # Gradient descent with momentum
            velocity = mom * velocity - lr * grad_current
            x_new = x_current + velocity
            
            # Keep within domain bounds
            x_new = np.clip(x_new, func_data['domain'][0], func_data['domain'][1])
            
            redraw_function()
        
        # Connect controls
        radio_functions.on_clicked(update_function)
        btn_run.on_clicked(run_optimization)
        btn_reset.on_clicked(reset_optimization)
        btn_step.on_clicked(single_step)
        
        # Update display when sliders change
        def update_display(val):
            redraw_function()
        
        slider_lr.on_changed(update_display)
        slider_momentum.on_changed(update_display)
        slider_start.on_changed(update_display)
        
        # Initial display
        redraw_function()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Gradient Descent Explorer loaded!")
        print("🎯 Try different functions and learning rates to see optimization behavior!")
        
        return fig

    def chain_rule_visualizer(self):
        """
        Interactive chain rule and backpropagation visualizer
        Shows how gradients flow through composed functions and neural networks
        """
        print("🎨 Starting Chain Rule & Backpropagation Visualizer...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('⛓️ Chain Rule & Neural Network Backpropagation', fontsize=16, color='cyan')
        
        # Demo 1: Simple function composition
        x_values = np.linspace(-2, 2, 100)
        
        def demo_chain_rule(x_input):
            """Demonstrate chain rule with f(g(x)) where g(x) = x^2, f(u) = sin(u)"""
            g_x = x_input**2
            f_g_x = np.sin(g_x)
            
            # Derivatives
            g_prime = 2 * x_input
            f_prime_at_g = np.cos(g_x)
            chain_rule_derivative = f_prime_at_g * g_prime
            
            return g_x, f_g_x, g_prime, f_prime_at_g, chain_rule_derivative
        
        # Interactive point for chain rule demo
        x_demo = 1.0
        g_val, fg_val, g_prime_val, f_prime_val, chain_val = demo_chain_rule(x_demo)
        
        # Plot function composition
        g_vals = x_values**2
        fg_vals = np.sin(g_vals)
        
        ax1.plot(x_values, g_vals, 'b-', label='g(x) = x²', linewidth=2)
        ax1.plot(x_demo, g_val, 'ro', markersize=8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('g(x)')
        ax1.set_title('Inner Function: g(x) = x²', color='lightblue')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(g_vals, fg_vals, 'r-', label='f(u) = sin(u)', linewidth=2)
        ax2.plot(g_val, fg_val, 'ro', markersize=8)
        ax2.set_xlabel('u = g(x)')
        ax2.set_ylabel('f(u)')
        ax2.set_title('Outer Function: f(u) = sin(u)', color='lightcoral')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Demo 2: Simple neural network
        def simple_neural_network(x_input, w1=1.5, w2=0.8, b1=0.5, b2=0.2):
            """Simple 2-layer network: x -> w1*x + b1 -> ReLU -> w2*h + b2"""
            # Forward pass
            z1 = w1 * x_input + b1
            h1 = np.maximum(0, z1)  # ReLU activation
            z2 = w2 * h1 + b2
            
            # For demonstration, let's say target is 2.0 and loss is (z2 - target)^2
            target = 2.0
            loss = (z2 - target)**2
            
            # Backward pass (gradients)
            dL_dz2 = 2 * (z2 - target)
            dL_dw2 = dL_dz2 * h1
            dL_dh1 = dL_dz2 * w2
            dL_dz1 = dL_dh1 * (1.0 if z1 > 0 else 0.0)  # ReLU derivative
            dL_dw1 = dL_dz1 * x_input
            dL_db1 = dL_dz1
            dL_db2 = dL_dz2
            
            return {
                'forward': {'z1': z1, 'h1': h1, 'z2': z2, 'loss': loss},
                'backward': {'dL_dz2': dL_dz2, 'dL_dw2': dL_dw2, 'dL_dh1': dL_dh1, 
                           'dL_dz1': dL_dz1, 'dL_dw1': dL_dw1, 'dL_db1': dL_db1, 'dL_db2': dL_db2}
            }
        
        # Neural network visualization
        x_input = 1.0
        network_result = simple_neural_network(x_input)
        
        # Draw network architecture
        ax3.clear()
        
        # Nodes
        input_pos = (0, 0.5)
        hidden_pos = (1, 0.5)
        output_pos = (2, 0.5)
        
        # Draw nodes
        ax3.add_patch(plt.Circle(input_pos, 0.1, color='lightblue', alpha=0.7))
        ax3.add_patch(plt.Circle(hidden_pos, 0.1, color='lightgreen', alpha=0.7))
        ax3.add_patch(plt.Circle(output_pos, 0.1, color='lightcoral', alpha=0.7))
        
        # Draw connections
        ax3.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]], 'w-', linewidth=2)
        ax3.plot([hidden_pos[0], output_pos[0]], [hidden_pos[1], output_pos[1]], 'w-', linewidth=2)
        
        # Labels
        ax3.text(input_pos[0], input_pos[1]-0.2, f'x = {x_input}', ha='center', color='white')
        ax3.text(hidden_pos[0], hidden_pos[1]-0.2, f'h = {network_result["forward"]["h1"]:.2f}', ha='center', color='white')
        ax3.text(output_pos[0], output_pos[1]-0.2, f'y = {network_result["forward"]["z2"]:.2f}', ha='center', color='white')
        
        # Weight labels
        ax3.text(0.5, 0.6, f'w₁ = 1.5', ha='center', color='yellow')
        ax3.text(1.5, 0.6, f'w₂ = 0.8', ha='center', color='yellow')
        
        ax3.set_xlim(-0.5, 2.5)
        ax3.set_ylim(0, 1)
        ax3.set_aspect('equal')
        ax3.set_title('Neural Network Forward Pass', color='lightgreen')
        ax3.axis('off')
        
        # Gradient flow visualization
        ax4.clear()
        
        # Same network structure
        ax4.add_patch(plt.Circle(input_pos, 0.1, color='lightblue', alpha=0.7))
        ax4.add_patch(plt.Circle(hidden_pos, 0.1, color='lightgreen', alpha=0.7))
        ax4.add_patch(plt.Circle(output_pos, 0.1, color='lightcoral', alpha=0.7))
        
        # Gradient flow arrows (backward)
        ax4.arrow(output_pos[0]-0.1, output_pos[1], -0.3, 0, head_width=0.05, head_length=0.05,
                 fc='red', ec='red', linewidth=2, alpha=0.8)
        ax4.arrow(hidden_pos[0]-0.1, hidden_pos[1], -0.3, 0, head_width=0.05, head_length=0.05,
                 fc='red', ec='red', linewidth=2, alpha=0.8)
        
        # Gradient labels
        ax4.text(1.5, 0.3, f'∂L/∂w₂ = {network_result["backward"]["dL_dw2"]:.2f}', ha='center', color='red')
        ax4.text(0.5, 0.3, f'∂L/∂w₁ = {network_result["backward"]["dL_dw1"]:.2f}', ha='center', color='red')
        ax4.text(output_pos[0], output_pos[1]+0.2, f'Loss = {network_result["forward"]["loss"]:.2f}', ha='center', color='red')
        
        ax4.set_xlim(-0.5, 2.5)
        ax4.set_ylim(0, 1)
        ax4.set_aspect('equal')
        ax4.set_title('Neural Network Backward Pass (Gradients)', color='lightcoral')
        ax4.axis('off')
        
        # Create sliders for interactive exploration
        plt.subplots_adjust(bottom=0.25)
        
        ax_x_slider = plt.axes([0.1, 0.15, 0.3, 0.03])
        ax_w1_slider = plt.axes([0.1, 0.11, 0.3, 0.03])
        ax_w2_slider = plt.axes([0.1, 0.07, 0.3, 0.03])
        
        slider_x = Slider(ax_x_slider, 'Input x', -2.0, 2.0, valinit=x_input, valfmt='%.2f')
        slider_w1 = Slider(ax_w1_slider, 'Weight w₁', 0.1, 3.0, valinit=1.5, valfmt='%.2f')
        slider_w2 = Slider(ax_w2_slider, 'Weight w₂', 0.1, 3.0, valinit=0.8, valfmt='%.2f')
        
        # Info display
        info_text = fig.text(0.55, 0.15, '', fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        def update_visualizations(val):
            """Update all visualizations when parameters change"""
            nonlocal x_demo
            
            # Update chain rule demo
            x_demo = slider_x.val
            g_val, fg_val, g_prime_val, f_prime_val, chain_val = demo_chain_rule(x_demo)
            
            # Update neural network
            network_result = simple_neural_network(slider_x.val, slider_w1.val, slider_w2.val)
            
            # Redraw plots
            ax1.clear()
            ax1.plot(x_values, x_values**2, 'b-', label='g(x) = x²', linewidth=2)
            ax1.plot(x_demo, g_val, 'ro', markersize=8)
            ax1.set_xlabel('x')
            ax1.set_ylabel('g(x)')
            ax1.set_title('Inner Function: g(x) = x²', color='lightblue')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.clear()
            u_values = np.linspace(0, 4, 100)
            ax2.plot(u_values, np.sin(u_values), 'r-', label='f(u) = sin(u)', linewidth=2)
            ax2.plot(g_val, fg_val, 'ro', markersize=8)
            ax2.set_xlabel('u = g(x)')
            ax2.set_ylabel('f(u)')
            ax2.set_title('Outer Function: f(u) = sin(u)', color='lightcoral')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Update neural network visualization
            ax3.clear()
            ax3.add_patch(plt.Circle(input_pos, 0.1, color='lightblue', alpha=0.7))
            ax3.add_patch(plt.Circle(hidden_pos, 0.1, color='lightgreen', alpha=0.7))
            ax3.add_patch(plt.Circle(output_pos, 0.1, color='lightcoral', alpha=0.7))
            ax3.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]], 'w-', linewidth=2)
            ax3.plot([hidden_pos[0], output_pos[0]], [hidden_pos[1], output_pos[1]], 'w-', linewidth=2)
            
            ax3.text(input_pos[0], input_pos[1]-0.2, f'x = {slider_x.val:.1f}', ha='center', color='white')
            ax3.text(hidden_pos[0], hidden_pos[1]-0.2, f'h = {network_result["forward"]["h1"]:.2f}', ha='center', color='white')
            ax3.text(output_pos[0], output_pos[1]-0.2, f'y = {network_result["forward"]["z2"]:.2f}', ha='center', color='white')
            ax3.text(0.5, 0.6, f'w₁ = {slider_w1.val:.1f}', ha='center', color='yellow')
            ax3.text(1.5, 0.6, f'w₂ = {slider_w2.val:.1f}', ha='center', color='yellow')
            
            ax3.set_xlim(-0.5, 2.5)
            ax3.set_ylim(0, 1)
            ax3.set_aspect('equal')
            ax3.set_title('Neural Network Forward Pass', color='lightgreen')
            ax3.axis('off')
            
            # Update gradient visualization
            ax4.clear()
            ax4.add_patch(plt.Circle(input_pos, 0.1, color='lightblue', alpha=0.7))
            ax4.add_patch(plt.Circle(hidden_pos, 0.1, color='lightgreen', alpha=0.7))
            ax4.add_patch(plt.Circle(output_pos, 0.1, color='lightcoral', alpha=0.7))
            
            ax4.arrow(output_pos[0]-0.1, output_pos[1], -0.3, 0, head_width=0.05, head_length=0.05,
                     fc='red', ec='red', linewidth=2, alpha=0.8)
            ax4.arrow(hidden_pos[0]-0.1, hidden_pos[1], -0.3, 0, head_width=0.05, head_length=0.05,
                     fc='red', ec='red', linewidth=2, alpha=0.8)
            
            ax4.text(1.5, 0.3, f'∂L/∂w₂ = {network_result["backward"]["dL_dw2"]:.2f}', ha='center', color='red')
            ax4.text(0.5, 0.3, f'∂L/∂w₁ = {network_result["backward"]["dL_dw1"]:.2f}', ha='center', color='red')
            ax4.text(output_pos[0], output_pos[1]+0.2, f'Loss = {network_result["forward"]["loss"]:.2f}', ha='center', color='red')
            
            ax4.set_xlim(-0.5, 2.5)
            ax4.set_ylim(0, 1)
            ax4.set_aspect('equal')
            ax4.set_title('Neural Network Backward Pass (Gradients)', color='lightcoral')
            ax4.axis('off')
            
            # Update info display
            info = f'''⛓️ Chain Rule Demo:
h(x) = sin(x²)
h'(x) = cos(x²) × 2x = {chain_val:.3f}

At x = {x_demo:.2f}:
• g(x) = x² = {g_val:.3f}
• f(g(x)) = sin(g(x)) = {fg_val:.3f}
• g'(x) = 2x = {g_prime_val:.3f}
• f'(g(x)) = cos(g(x)) = {f_prime_val:.3f}
• h'(x) = f'(g(x)) × g'(x) = {chain_val:.3f}

🧠 Neural Network:
Forward: x → z₁ = w₁x + b₁ → h₁ = ReLU(z₁) → z₂ = w₂h₁ + b₂
Backward: ∂L/∂w₁ ← ∂L/∂z₁ ← ∂L/∂h₁ ← ∂L/∂z₂

Target = 2.0, Loss = (output - target)²
Current Loss = {network_result["forward"]["loss"]:.3f}'''
            
            info_text.set_text(info)
            plt.draw()
        
        # Connect sliders
        slider_x.on_changed(update_visualizations)
        slider_w1.on_changed(update_visualizations)
        slider_w2.on_changed(update_visualizations)
        
        # Initial update
        update_visualizations(None)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Chain Rule Visualizer loaded!")
        print("🎯 Adjust parameters to see how gradients flow through functions and networks!")
        
        return fig

    def optimization_landscape_3d(self):
        """
        3D optimization landscape explorer
        Shows gradient descent on complex 2D functions
        """
        print("🎨 Starting 3D Optimization Landscape Explorer...")
        
        fig = plt.figure(figsize=(14, 8))
        
        # Create 3D landscape plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        
        fig.suptitle('🏔️ 3D Optimization Landscapes', fontsize=16, color='cyan')
        
        # Define 2D test functions
        functions_2d = {
            'Simple Bowl': {
                'func': lambda x, y: (x-1)**2 + (y-2)**2,
                'grad': lambda x, y: (2*(x-1), 2*(y-2)),
                'range': (-2, 4, -1, 5),
                'minimum': (1, 2)
            },
            'Rosenbrock Valley': {
                'func': lambda x, y: (1-x)**2 + 100*(y-x**2)**2,
                'grad': lambda x, y: (-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2)),
                'range': (-2, 2, -1, 3),
                'minimum': (1, 1)
            },
            'Himmelblau Function': {
                'func': lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
                'grad': lambda x, y: (4*x*(x**2 + y - 11) + 2*(x + y**2 - 7), 
                                    2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)),
                'range': (-5, 5, -5, 5),
                'minimum': 'Multiple'
            },
            'Saddle Point': {
                'func': lambda x, y: x**2 - y**2,
                'grad': lambda x, y: (2*x, -2*y),
                'range': (-3, 3, -3, 3),
                'minimum': (0, 0)
            }
        }
        
        current_func = 'Simple Bowl'
        func_data = functions_2d[current_func]
        
        # Create meshgrid for plotting
        x_range = func_data['range']
        x = np.linspace(x_range[0], x_range[1], 50)
        y = np.linspace(x_range[2], x_range[3], 50)
        X, Y = np.meshgrid(x, y)
        Z = func_data['func'](X, Y)
        
        # Plot 3D surface
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, antialiased=True)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        ax1.set_title(f'3D Landscape: {current_func}', color='lightblue')
        
        # Plot 2D contour
        contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Contour Plot with Optimization Path', color='lightgreen')
        ax2.grid(True, alpha=0.3)
        
        # Optimization parameters
        learning_rate = 0.01
        momentum = 0.9
        start_point = (x_range[0] + 0.5, x_range[2] + 0.5)
        
        # Track optimization path
        optimization_path_3d = []
        
        def run_gradient_descent_2d():
            """Run 2D gradient descent optimization"""
            nonlocal optimization_path_3d
            
            # Reset path
            optimization_path_3d = []
            velocity_x, velocity_y = 0.0, 0.0
            
            x_current, y_current = start_point
            max_steps = 500
            tolerance = 1e-6
            
            for step in range(max_steps):
                # Record current position
                z_current = func_data['func'](x_current, y_current)
                optimization_path_3d.append((x_current, y_current, z_current))
                
                # Compute gradient
                grad_x, grad_y = func_data['grad'](x_current, y_current)
                
                # Check convergence
                if abs(grad_x) < tolerance and abs(grad_y) < tolerance:
                    print(f"✅ Converged in {step+1} steps at ({x_current:.4f}, {y_current:.4f})")
                    break
                
                # Update with momentum
                velocity_x = momentum * velocity_x - learning_rate * grad_x
                velocity_y = momentum * velocity_y - learning_rate * grad_y
                
                x_current += velocity_x
                y_current += velocity_y
                
                # Keep within bounds
                x_current = np.clip(x_current, x_range[0], x_range[1])
                y_current = np.clip(y_current, x_range[2], x_range[3])
            
            return optimization_path_3d
        
        def update_visualization():
            """Update the 3D and 2D visualizations"""
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Recompute surface
            Z = func_data['func'](X, Y)
            
            # Plot 3D surface
            surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, antialiased=True)
            
            # Plot optimization path in 3D
            if optimization_path_3d:
                path_x = [p[0] for p in optimization_path_3d]
                path_y = [p[1] for p in optimization_path_3d]
                path_z = [p[2] for p in optimization_path_3d]
                
                ax1.plot(path_x, path_y, path_z, 'r-', linewidth=3, alpha=0.8, label='Optimization Path')
                ax1.scatter(path_x[0], path_y[0], path_z[0], color='green', s=100, label='Start')
                ax1.scatter(path_x[-1], path_y[-1], path_z[-1], color='blue', s=100, label='End')
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('f(x,y)')
            ax1.set_title(f'3D Landscape: {current_func}', color='lightblue')
            ax1.legend()
            
            # Plot 2D contour
            contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
            ax2.clabel(contour, inline=True, fontsize=8)
            
            # Plot optimization path in 2D
            if optimization_path_3d:
                path_x = [p[0] for p in optimization_path_3d]
                path_y = [p[1] for p in optimization_path_3d]
                
                ax2.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, label='Optimization Path')
                ax2.scatter(path_x[0], path_y[0], color='green', s=100, label='Start')
                ax2.scatter(path_x[-1], path_y[-1], color='blue', s=100, label='End')
                
                # Plot gradient arrows at selected points
                n_arrows = min(10, len(path_x))
                for i in range(0, len(path_x)-1, max(1, len(path_x)//n_arrows)):
                    grad_x, grad_y = func_data['grad'](path_x[i], path_y[i])
                    arrow_scale = 0.1
                    ax2.arrow(path_x[i], path_y[i], -arrow_scale*grad_x, -arrow_scale*grad_y,
                             head_width=0.1, head_length=0.1, fc='yellow', ec='yellow', alpha=0.7)
            
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title('Contour Plot with Optimization Path', color='lightgreen')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.draw()
        
        # Create controls
        plt.subplots_adjust(bottom=0.25)
        
        # Function selection buttons
        ax_buttons = plt.axes([0.1, 0.15, 0.3, 0.08])
        func_buttons = RadioButtons(ax_buttons, list(functions_2d.keys()))
        
        # Parameter sliders
        ax_lr = plt.axes([0.1, 0.08, 0.3, 0.03])
        ax_momentum = plt.axes([0.1, 0.04, 0.3, 0.03])
        
        slider_lr = Slider(ax_lr, 'Learning Rate', 0.001, 0.1, valinit=learning_rate, valfmt='%.3f')
        slider_momentum = Slider(ax_momentum, 'Momentum', 0.0, 0.99, valinit=momentum, valfmt='%.2f')
        
        # Control buttons
        btn_optimize = Button(plt.axes([0.5, 0.12, 0.1, 0.04]), 'Optimize', color='green')
        btn_reset = Button(plt.axes([0.5, 0.08, 0.1, 0.04]), 'Reset', color='red')
        
        def change_function(label):
            """Change the optimization function"""
            nonlocal current_func, func_data, X, Y, start_point
            current_func = label
            func_data = functions_2d[current_func]
            
            # Update meshgrid
            x_range = func_data['range']
            x = np.linspace(x_range[0], x_range[1], 50)
            y = np.linspace(x_range[2], x_range[3], 50)
            X, Y = np.meshgrid(x, y)
            
            # Update start point
            start_point = (x_range[0] + 0.5, x_range[2] + 0.5)
            
            update_visualization()
        
        def optimize(event):
            """Run optimization"""
            nonlocal learning_rate, momentum
            learning_rate = slider_lr.val
            momentum = slider_momentum.val
            run_gradient_descent_2d()
            update_visualization()
        
        def reset(event):
            """Reset optimization"""
            nonlocal optimization_path_3d
            optimization_path_3d = []
            update_visualization()
        
        # Connect controls
        func_buttons.on_clicked(change_function)
        btn_optimize.on_clicked(optimize)
        btn_reset.on_clicked(reset)
        
        # Initial visualization
        update_visualization()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ 3D Optimization Landscape loaded!")
        print("🎯 Explore different functions and see how optimization navigates complex landscapes!")
        
        return fig

    def neural_network_trainer(self):
        """
        Interactive neural network training demonstration
        Shows real-time loss reduction and weight updates
        """
        print("🎨 Starting Neural Network Training Demo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🧠 Neural Network Training in Real-Time', fontsize=16, color='cyan')
        
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 100
        X_data = np.random.randn(n_samples, 2)  # 2D input
        y_data = X_data[:, 0]**2 + X_data[:, 1]**2 + 0.1*np.random.randn(n_samples)  # Quadratic target + noise
        
        # Split into train/test
        train_size = int(0.8 * n_samples)
        X_train, X_test = X_data[:train_size], X_data[train_size:]
        y_train, y_test = y_data[:train_size], y_data[train_size:]
        
        # Neural network implementation
        class SimpleNN:
            def __init__(self, input_size=2, hidden_size=5, output_size=1):
                # Initialize weights
                self.W1 = np.random.randn(input_size, hidden_size) * 0.5
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, output_size) * 0.5
                self.b2 = np.zeros((1, output_size))
                
                # Track training history
                self.loss_history = []
                self.weight_history = []
                
            def relu(self, x):
                return np.maximum(0, x)
            
            def relu_derivative(self, x):
                return (x > 0).astype(float)
            
            def forward(self, X):
                self.z1 = np.dot(X, self.W1) + self.b1
                self.a1 = self.relu(self.z1)
                self.z2 = np.dot(self.a1, self.W2) + self.b2
                return self.z2
            
            def backward(self, X, y, output):
                m = X.shape[0]
                
                # Output layer gradients
                dz2 = output - y.reshape(-1, 1)
                dW2 = (1/m) * np.dot(self.a1.T, dz2)
                db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
                
                # Hidden layer gradients
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.relu_derivative(self.z1)
                dW1 = (1/m) * np.dot(X.T, dz1)
                db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
                
                return dW1, db1, dW2, db2
            
            def update_weights(self, dW1, db1, dW2, db2, learning_rate):
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
            
            def train_step(self, X, y, learning_rate=0.01):
                # Forward pass
                output = self.forward(X)
                
                # Compute loss (MSE)
                loss = np.mean((output.flatten() - y)**2)
                self.loss_history.append(loss)
                
                # Backward pass
                dW1, db1, dW2, db2 = self.backward(X, y, output)
                
                # Update weights
                self.update_weights(dW1, db1, dW2, db2, learning_rate)
                
                # Store weight snapshot
                self.weight_history.append({
                    'W1': self.W1.copy(),
                    'W2': self.W2.copy(),
                    'loss': loss
                })
                
                return loss
        
        # Initialize network
        nn = SimpleNN()
        
        # Training parameters
        learning_rate = 0.01
        current_epoch = 0
        max_epochs = 1000
        
        # Plot initial data
        ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('Training Data (color = target value)', color='lightblue')
        ax1.grid(True, alpha=0.3)
        
        # Initialize plots
        loss_line, = ax2.plot([], [], 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss Over Time', color='lightgreen')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        
        # Weight visualization
        im_W1 = ax3.imshow(nn.W1.T, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
        ax3.set_title('Hidden Layer Weights (W1)', color='lightyellow')
        ax3.set_xlabel('Input Features')
        ax3.set_ylabel('Hidden Neurons')
        plt.colorbar(im_W1, ax=ax3, fraction=0.046, pad=0.04)
        
        # Prediction surface
        xx, yy = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = nn.forward(grid_points).reshape(xx.shape)
        
        contour = ax4.contourf(xx, yy, predictions, levels=20, cmap='viridis', alpha=0.7)
        ax4.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', 
                   edgecolors='white', s=50, alpha=0.9)
        ax4.set_xlabel('Feature 1')
        ax4.set_ylabel('Feature 2')
        ax4.set_title('Learned Function (test data overlaid)', color='lightcoral')
        plt.colorbar(contour, ax=ax4, fraction=0.046, pad=0.04)
        
        # Create controls
        plt.subplots_adjust(bottom=0.25)
        
        # Sliders
        ax_lr = plt.axes([0.1, 0.15, 0.3, 0.03])
        slider_lr = Slider(ax_lr, 'Learning Rate', 0.001, 0.1, valinit=learning_rate, valfmt='%.3f')
        
        # Buttons
        btn_train_step = Button(plt.axes([0.5, 0.15, 0.08, 0.04]), 'Train Step', color='green')
        btn_train_epoch = Button(plt.axes([0.6, 0.15, 0.08, 0.04]), 'Train Epoch', color='blue')
        btn_reset = Button(plt.axes([0.7, 0.15, 0.08, 0.04]), 'Reset', color='red')
        
        # Info display
        info_text = fig.text(0.1, 0.08, '', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        def update_visualizations():
            """Update all visualization panels"""
            # Update loss plot
            if nn.loss_history:
                epochs = list(range(len(nn.loss_history)))
                loss_line.set_data(epochs, nn.loss_history)
                ax2.set_xlim(0, max(100, len(nn.loss_history)))
                ax2.set_ylim(0, max(1, max(nn.loss_history)))
                ax2.draw_artist(loss_line)
            
            # Update weight visualization
            im_W1.set_array(nn.W1.T)
            im_W1.set_clim(vmin=nn.W1.min(), vmax=nn.W1.max())
            
            # Update prediction surface
            predictions = nn.forward(grid_points).reshape(xx.shape)
            ax4.clear()
            contour = ax4.contourf(xx, yy, predictions, levels=20, cmap='viridis', alpha=0.7)
            ax4.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',
                       edgecolors='white', s=50, alpha=0.9)
            ax4.set_xlabel('Feature 1')
            ax4.set_ylabel('Feature 2')
            ax4.set_title('Learned Function (test data overlaid)', color='lightcoral')
            
            # Update info
            if nn.loss_history:
                current_loss = nn.loss_history[-1]
                test_predictions = nn.forward(X_test)
                test_loss = np.mean((test_predictions.flatten() - y_test)**2)
                
                info = f'''🧠 Neural Network Training Status:
Epochs Completed: {len(nn.loss_history)}
Current Training Loss: {current_loss:.4f}
Current Test Loss: {test_loss:.4f}
Learning Rate: {slider_lr.val:.3f}

Network Architecture:
• Input: 2 features
• Hidden: 5 neurons (ReLU)
• Output: 1 value
• Parameters: {2*5 + 5 + 5*1 + 1} total

🎯 Optimization Progress:
• Gradient descent updates all weights
• Loss decreases → network is learning
• Watch prediction surface adapt to data'''
                info_text.set_text(info)
            
            plt.draw()
        
        def train_single_step(event):
            """Train for one step"""
            nonlocal learning_rate
            learning_rate = slider_lr.val
            loss = nn.train_step(X_train, y_train, learning_rate)
            update_visualizations()
        
        def train_epoch(event):
            """Train for one full epoch (10 steps)"""
            nonlocal learning_rate
            learning_rate = slider_lr.val
            for _ in range(10):
                nn.train_step(X_train, y_train, learning_rate)
            update_visualizations()
        
        def reset_network(event):
            """Reset the neural network"""
            nonlocal nn
            nn = SimpleNN()
            update_visualizations()
        
        # Connect controls
        btn_train_step.on_clicked(train_single_step)
        btn_train_epoch.on_clicked(train_epoch)
        btn_reset.on_clicked(reset_network)
        
        # Initial update
        update_visualizations()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Neural Network Trainer loaded!")
        print("🎯 Watch the network learn in real-time through gradient descent!")
        
        return fig

    def run_comprehensive_demo(self):
        """
        Run all calculus visualizations in sequence
        Complete visual learning experience for Week 2
        """
        print("🚀 Starting Comprehensive Calculus & Optimization Journey!")
        print("=" * 60)
        
        demos = [
            ("Gradient Descent Explorer", self.gradient_descent_explorer),
            ("Chain Rule & Backpropagation", self.chain_rule_visualizer),
            ("3D Optimization Landscapes", self.optimization_landscape_3d),
            ("Neural Network Training", self.neural_network_trainer)
        ]
        
        print("📋 Available Demos:")
        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        
        print("\n🎮 Interactive Demo Features:")
        print("  • Real-time gradient descent optimization")
        print("  • Learning rate and momentum parameter tuning")
        print("  • Chain rule visualization for neural networks")
        print("  • 3D landscape exploration with contour plots")
        print("  • Live neural network training with loss tracking")
        
        # Run each demo
        figures = []
        for name, demo_func in demos:
            print(f"\n🎨 Loading {name}...")
            try:
                fig = demo_func()
                figures.append((name, fig))
                print(f"✅ {name} ready!")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 All calculus demos loaded successfully!")
        print("🎯 Key Learning Objectives Achieved:")
        print("  ✅ Gradient descent optimization in various landscapes")
        print("  ✅ Chain rule enabling backpropagation in neural networks")
        print("  ✅ Learning rate effects on convergence and stability")
        print("  ✅ Real-time neural network training visualization")
        print("  ✅ Connection between calculus theory and AI practice")
        print("\n🚀 Ready for Week 3: Probability & Statistics!")
        print("💡 You now understand the optimization engine of AI!")
        
        return figures

# Convenience functions for direct execution
def gradient_descent_explorer():
    """Quick access to gradient descent explorer"""
    viz = CalculusVisualizer()
    return viz.gradient_descent_explorer()

def chain_rule_visualizer():
    """Quick access to chain rule and backpropagation visualizer"""
    viz = CalculusVisualizer()
    return viz.chain_rule_visualizer()

def optimization_landscape_3d():
    """Quick access to 3D optimization landscape explorer"""
    viz = CalculusVisualizer()
    return viz.optimization_landscape_3d()

def neural_network_trainer():
    """Quick access to neural network training demo"""
    viz = CalculusVisualizer()
    return viz.neural_network_trainer()

def run_all_demos():
    """Run complete calculus learning experience"""
    viz = CalculusVisualizer()
    return viz.run_comprehensive_demo()

# Main execution for browser environment
if __name__ == "__main__":
    print("🧠 Neural Odyssey - Week 2: Calculus & Optimization Visualizations")
    print("=" * 70)
    print("🎨 Available Functions:")
    print("  • gradient_descent_explorer() - Interactive optimization landscapes")
    print("  • chain_rule_visualizer() - Backpropagation and gradient flow")
    print("  • optimization_landscape_3d() - 3D landscape exploration")
    print("  • neural_network_trainer() - Real-time neural network training")
    print("  • run_all_demos() - Complete calculus learning experience")
    print("\n🚀 Type any function name to start!")
    print("💡 Example: run_all_demos()")
    print("\n🎯 This week: Learn how AI systems optimize and improve!")
    
    # For automatic execution in browser
    # Uncomment the next line to auto-run all demos
    # run_all_demos()