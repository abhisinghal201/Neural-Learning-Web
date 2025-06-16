"""
Neural Odyssey - Week 1: Linear Algebra Interactive Visualizations
Browser-executable Python code for visual learning and interactive demos

This module provides interactive visualizations for understanding:
- Vector operations and geometric interpretations
- Matrix transformations and their effects on 2D shapes
- Eigenvalues and eigenvectors as special directions
- Real-world applications like PageRank algorithm
- Data analysis through linear algebra lens

Designed for browser execution via Pyodide with full interactivity.
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
import math
from typing import Tuple, List, Optional

# Configure matplotlib for better browser display
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#2d2d2d'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

class LinearAlgebraVisualizer:
    """Main class for linear algebra visualizations"""
    
    def __init__(self):
        self.current_demo = None
        self.animation = None
        
    def create_matrix_transformer(self):
        """
        Interactive matrix transformation visualizer
        Shows how 2x2 matrices transform shapes in real-time
        """
        print("🎨 Starting Matrix Transformation Visualizer...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('🔄 Matrix Transformation Visualizer', fontsize=16, color='cyan')
        
        # Initial shape (unit square)
        original_shape = np.array([[0, 1, 1, 0, 0],
                                  [0, 0, 1, 1, 0]])
        
        # Initial transformation matrix (identity)
        transform_matrix = np.array([[1.0, 0.0],
                                   [0.0, 1.0]])
        
        # Plot original shape
        ax1.plot(original_shape[0], original_shape[1], 'g-', linewidth=3, label='Original')
        ax1.fill(original_shape[0], original_shape[1], 'green', alpha=0.3)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Original Shape', color='lightgreen')
        ax1.legend()
        
        # Plot transformed shape
        transformed_shape = transform_matrix @ original_shape
        line2, = ax2.plot(transformed_shape[0], transformed_shape[1], 'r-', linewidth=3, label='Transformed')
        fill2 = ax2.fill(transformed_shape[0], transformed_shape[1], 'red', alpha=0.3)
        
        # Add grid and eigenvector visualization
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Transformed Shape', color='lightcoral')
        
        # Create sliders for matrix elements
        plt.subplots_adjust(bottom=0.25)
        
        # Slider axes
        ax_a = plt.axes([0.1, 0.15, 0.3, 0.03])
        ax_b = plt.axes([0.1, 0.11, 0.3, 0.03])  
        ax_c = plt.axes([0.1, 0.07, 0.3, 0.03])
        ax_d = plt.axes([0.1, 0.03, 0.3, 0.03])
        
        # Create sliders
        slider_a = Slider(ax_a, 'Matrix[0,0]', -2.0, 2.0, valinit=1.0, valfmt='%.1f')
        slider_b = Slider(ax_b, 'Matrix[0,1]', -2.0, 2.0, valinit=0.0, valfmt='%.1f')
        slider_c = Slider(ax_c, 'Matrix[1,0]', -2.0, 2.0, valinit=0.0, valfmt='%.1f')
        slider_d = Slider(ax_d, 'Matrix[1,1]', -2.0, 2.0, valinit=1.0, valfmt='%.1f')
        
        # Matrix display
        matrix_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, 
                              fontsize=12, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Eigenvalue/eigenvector display
        eigen_text = ax2.text(0.02, 0.02, '', transform=ax2.transAxes,
                             fontsize=10, verticalalignment='bottom',
                             bbox=dict(boxstyle='round', facecolor='navy', alpha=0.8))
        
        def update_transformation(val):
            """Update the transformation when sliders change"""
            nonlocal transform_matrix, line2, fill2
            
            # Get new matrix values
            a, b, c, d = slider_a.val, slider_b.val, slider_c.val, slider_d.val
            transform_matrix = np.array([[a, b], [c, d]])
            
            # Apply transformation
            transformed_shape = transform_matrix @ original_shape
            
            # Update plot
            line2.set_data(transformed_shape[0], transformed_shape[1])
            
            # Clear and redraw fill
            for collection in ax2.collections:
                collection.remove()
            ax2.fill(transformed_shape[0], transformed_shape[1], 'red', alpha=0.3)
            
            # Update matrix display
            det = np.linalg.det(transform_matrix)
            matrix_text.set_text(f'Matrix:\n[{a:.1f}  {b:.1f}]\n[{c:.1f}  {d:.1f}]\n\nDeterminant: {det:.2f}')
            
            # Calculate and display eigenvalues/eigenvectors
            try:
                eigenvals, eigenvecs = np.linalg.eig(transform_matrix)
                eigen_info = f'Eigenvalues:\nλ₁ = {eigenvals[0]:.2f}\nλ₂ = {eigenvals[1]:.2f}'
                eigen_text.set_text(eigen_info)
                
                # Draw eigenvectors if they're real
                ax2.clear()
                ax2.plot(transformed_shape[0], transformed_shape[1], 'r-', linewidth=3, label='Transformed')
                ax2.fill(transformed_shape[0], transformed_shape[1], 'red', alpha=0.3)
                
                for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
                    if np.isreal(val) and np.isreal(vec).all():
                        vec = vec.real
                        val = val.real
                        color = 'yellow' if i == 0 else 'orange'
                        ax2.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1, 
                                fc=color, ec=color, linewidth=2, label=f'Eigenvector {i+1}')
                        ax2.arrow(0, 0, -vec[0], -vec[1], head_width=0.1, head_length=0.1,
                                fc=color, ec=color, linewidth=2, alpha=0.7)
                
                ax2.set_xlim(-3, 3)
                ax2.set_ylim(-3, 3)
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Transformed Shape + Eigenvectors', color='lightcoral')
                ax2.legend()
                
            except np.linalg.LinAlgError:
                eigen_text.set_text('Eigenvalues:\nSingular matrix!')
            
            plt.draw()
        
        # Connect sliders to update function
        slider_a.on_changed(update_transformation)
        slider_b.on_changed(update_transformation)
        slider_c.on_changed(update_transformation)
        slider_d.on_changed(update_transformation)
        
        # Add preset transformation buttons
        ax_presets = plt.axes([0.6, 0.15, 0.35, 0.08])
        presets = {
            'Identity': [1, 0, 0, 1],
            'Rotation 45°': [0.71, -0.71, 0.71, 0.71],
            'Scale 2x': [2, 0, 0, 2],
            'Shear X': [1, 0.5, 0, 1],
            'Reflection': [1, 0, 0, -1]
        }
        
        preset_buttons = {}
        for i, (name, values) in enumerate(presets.items()):
            button_ax = plt.axes([0.6 + (i % 3) * 0.12, 0.15 - (i // 3) * 0.04, 0.1, 0.03])
            button = Button(button_ax, name, color='darkblue', hovercolor='blue')
            
            def make_preset_callback(vals):
                def callback(event):
                    slider_a.set_val(vals[0])
                    slider_b.set_val(vals[1])
                    slider_c.set_val(vals[2])
                    slider_d.set_val(vals[3])
                return callback
            
            button.on_clicked(make_preset_callback(values))
            preset_buttons[name] = button
        
        # Initial update
        update_transformation(None)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Matrix Transformer loaded! Use sliders to see transformations in real-time.")
        print("🎯 Try the preset buttons to see classic transformations!")
        
        return fig

    def create_vector_playground(self):
        """
        Interactive vector operations playground
        Visualize vector addition, scalar multiplication, dot products
        """
        print("🎨 Starting Vector Operations Playground...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle('🧮 Vector Operations Playground', fontsize=16, color='cyan')
        
        # Initial vectors
        vector_a = np.array([2, 1])
        vector_b = np.array([1, 2])
        
        # Plot setup
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linewidth=0.5)
        ax.axvline(x=0, color='white', linewidth=0.5)
        
        # Vector plots
        arrow_a = ax.arrow(0, 0, vector_a[0], vector_a[1], head_width=0.2, head_length=0.2,
                          fc='red', ec='red', linewidth=3, label='Vector A')
        arrow_b = ax.arrow(0, 0, vector_b[0], vector_b[1], head_width=0.2, head_length=0.2,
                          fc='blue', ec='blue', linewidth=3, label='Vector B')
        
        # Result vector (initially sum)
        result_vector = vector_a + vector_b
        arrow_result = ax.arrow(0, 0, result_vector[0], result_vector[1], head_width=0.2, head_length=0.2,
                               fc='yellow', ec='yellow', linewidth=3, label='Result', linestyle='--')
        
        # Create sliders
        plt.subplots_adjust(bottom=0.3, right=0.8)
        
        # Vector A sliders
        ax_a_x = plt.axes([0.1, 0.2, 0.3, 0.03])
        ax_a_y = plt.axes([0.1, 0.16, 0.3, 0.03])
        slider_a_x = Slider(ax_a_x, 'Vector A X', -4.0, 4.0, valinit=2.0, valfmt='%.1f')
        slider_a_y = Slider(ax_a_y, 'Vector A Y', -4.0, 4.0, valinit=1.0, valfmt='%.1f')
        
        # Vector B sliders  
        ax_b_x = plt.axes([0.1, 0.12, 0.3, 0.03])
        ax_b_y = plt.axes([0.1, 0.08, 0.3, 0.03])
        slider_b_x = Slider(ax_b_x, 'Vector B X', -4.0, 4.0, valinit=1.0, valfmt='%.1f')
        slider_b_y = Slider(ax_b_y, 'Vector B Y', -4.0, 4.0, valinit=2.0, valfmt='%.1f')
        
        # Scalar slider
        ax_scalar = plt.axes([0.1, 0.04, 0.3, 0.03])
        slider_scalar = Slider(ax_scalar, 'Scalar', -3.0, 3.0, valinit=1.0, valfmt='%.1f')
        
        # Operation selection
        ax_ops = plt.axes([0.82, 0.6, 0.15, 0.3])
        operation_labels = ['A + B', 'A - B', 'Scalar × A', 'Dot Product', 'Cross Product']
        operations_check = CheckButtons(ax_ops, operation_labels, [True, False, False, False, False])
        
        # Info display
        info_text = ax.text(0.82, 0.5, '', transform=fig.transFigure, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        def update_vectors(val):
            """Update vector visualization"""
            nonlocal vector_a, vector_b, arrow_a, arrow_b, arrow_result
            
            # Get new vector values
            vector_a = np.array([slider_a_x.val, slider_a_y.val])
            vector_b = np.array([slider_b_x.val, slider_b_y.val])
            scalar = slider_scalar.val
            
            # Clear previous arrows
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='white', linewidth=0.5)
            ax.axvline(x=0, color='white', linewidth=0.5)
            
            # Draw vectors A and B
            ax.arrow(0, 0, vector_a[0], vector_a[1], head_width=0.2, head_length=0.2,
                    fc='red', ec='red', linewidth=3, label='Vector A')
            ax.arrow(0, 0, vector_b[0], vector_b[1], head_width=0.2, head_length=0.2,
                    fc='blue', ec='blue', linewidth=3, label='Vector B')
            
            # Determine operation and result
            active_ops = [label for label, active in zip(operation_labels, operations_check.get_status()) if active]
            
            if 'A + B' in active_ops:
                result = vector_a + vector_b
                ax.arrow(0, 0, result[0], result[1], head_width=0.2, head_length=0.2,
                        fc='yellow', ec='yellow', linewidth=3, label='A + B', linestyle='--')
                # Show parallelogram construction
                ax.arrow(vector_a[0], vector_a[1], vector_b[0], vector_b[1], 
                        head_width=0.15, head_length=0.15, fc='green', ec='green', alpha=0.7)
                ax.arrow(vector_b[0], vector_b[1], vector_a[0], vector_a[1],
                        head_width=0.15, head_length=0.15, fc='green', ec='green', alpha=0.7)
                
            if 'A - B' in active_ops:
                result = vector_a - vector_b  
                ax.arrow(0, 0, result[0], result[1], head_width=0.2, head_length=0.2,
                        fc='orange', ec='orange', linewidth=3, label='A - B', linestyle=':')
                
            if 'Scalar × A' in active_ops:
                result = scalar * vector_a
                ax.arrow(0, 0, result[0], result[1], head_width=0.2, head_length=0.2,
                        fc='purple', ec='purple', linewidth=3, label=f'{scalar:.1f} × A', linestyle='-.')
            
            # Calculate and display metrics
            magnitude_a = np.linalg.norm(vector_a)
            magnitude_b = np.linalg.norm(vector_b)
            dot_product = np.dot(vector_a, vector_b)
            
            # Angle between vectors
            cos_angle = dot_product / (magnitude_a * magnitude_b) if magnitude_a * magnitude_b != 0 else 0
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical safety
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            # Cross product (in 2D, it's the z-component)
            cross_2d = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
            
            # Update info display
            info = f'''Vector Metrics:
|A| = {magnitude_a:.2f}
|B| = {magnitude_b:.2f}

Dot Product:
A · B = {dot_product:.2f}

Angle:
θ = {angle_deg:.1f}°

Cross Product (2D):
A × B = {cross_2d:.2f}

Unit Vectors:
Â = ({vector_a[0]/magnitude_a:.2f}, {vector_a[1]/magnitude_a:.2f})
B̂ = ({vector_b[0]/magnitude_b:.2f}, {vector_b[1]/magnitude_b:.2f})'''
            
            info_text.set_text(info)
            
            ax.legend(loc='upper left')
            ax.set_title(f'Operations: {", ".join(active_ops)}', color='lightgreen')
            plt.draw()
        
        # Connect controls
        slider_a_x.on_changed(update_vectors)
        slider_a_y.on_changed(update_vectors)
        slider_b_x.on_changed(update_vectors)
        slider_b_y.on_changed(update_vectors)
        slider_scalar.on_changed(update_vectors)
        operations_check.on_clicked(lambda label: update_vectors(None))
        
        # Initial update
        update_vectors(None)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Vector Playground loaded! Experiment with different operations.")
        print("🎯 Try changing vectors and see how operations affect the results!")
        
        return fig

    def create_pagerank_visualizer(self):
        """
        PageRank algorithm visualization
        Real-world application of eigenvalues and eigenvectors
        """
        print("🎨 Starting PageRank Algorithm Visualizer...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('🔍 PageRank Algorithm: Eigenvalues in Action', fontsize=16, color='cyan')
        
        # Create sample web graph
        n_pages = 5
        pages = ['Home', 'About', 'Blog', 'Contact', 'Products']
        
        # Adjacency matrix (who links to whom)
        # Example: Home links to About and Blog, About links to Home and Contact, etc.
        adjacency = np.array([
            [0, 1, 1, 0, 1],  # Home -> About, Blog, Products
            [1, 0, 0, 1, 0],  # About -> Home, Contact  
            [1, 1, 0, 0, 1],  # Blog -> Home, About, Products
            [0, 1, 0, 0, 0],  # Contact -> About
            [1, 0, 1, 1, 0]   # Products -> Home, Blog, Contact
        ])
        
        # Convert to Google matrix (column stochastic)
        damping_factor = 0.85
        google_matrix = self._create_google_matrix(adjacency, damping_factor)
        
        # Visualize the web graph
        pos = {
            0: (0.5, 0.8),    # Home (top center)
            1: (0.2, 0.5),    # About (left)
            2: (0.8, 0.5),    # Blog (right)
            3: (0.2, 0.2),    # Contact (bottom left)
            4: (0.8, 0.2)     # Products (bottom right)
        }
        
        # Draw nodes
        for i, (page, (x, y)) in enumerate(zip(pages, pos.values())):
            circle = plt.Circle((x, y), 0.08, color='lightblue', alpha=0.7)
            ax1.add_patch(circle)
            ax1.text(x, y, page, ha='center', va='center', fontsize=8, weight='bold')
        
        # Draw edges (links)
        for i in range(n_pages):
            for j in range(n_pages):
                if adjacency[i, j] > 0:
                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    # Add slight curve to avoid overlapping arrows
                    if i != j:
                        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                   arrowprops=dict(arrowstyle='->', color='yellow', lw=2, alpha=0.7))
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_title('Web Graph Structure', color='lightblue')
        ax1.axis('off')
        
        # PageRank iteration visualization
        initial_rank = np.ones(n_pages) / n_pages  # Equal initial ranking
        current_rank = initial_rank.copy()
        
        # Store iteration history
        rank_history = [current_rank.copy()]
        
        # Perform PageRank iterations
        max_iterations = 20
        for iteration in range(max_iterations):
            new_rank = google_matrix @ current_rank
            rank_history.append(new_rank.copy())
            
            # Check convergence
            if np.allclose(current_rank, new_rank, atol=1e-6):
                print(f"📊 PageRank converged after {iteration + 1} iterations")
                break
            current_rank = new_rank
        
        # Plot convergence
        iterations = range(len(rank_history))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, page in enumerate(pages):
            page_ranks = [ranks[i] for ranks in rank_history]
            ax2.plot(iterations, page_ranks, 'o-', color=colors[i], 
                    linewidth=2, markersize=4, label=page)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PageRank Score')
        ax2.set_title('PageRank Convergence', color='lightgreen')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate and display final eigenvalue analysis
        eigenvals, eigenvecs = np.linalg.eig(google_matrix)
        dominant_eigenval = eigenvals[0]
        dominant_eigenvec = np.abs(eigenvecs[:, 0])
        dominant_eigenvec = dominant_eigenvec / np.sum(dominant_eigenvec)  # Normalize
        
        # Add final rankings text
        final_ranks = current_rank
        ranking_text = "🏆 Final PageRank Scores:\n"
        for i, (page, score) in enumerate(zip(pages, final_ranks)):
            ranking_text += f"{i+1}. {page}: {score:.3f}\n"
        
        ranking_text += f"\n🔬 Mathematical Insight:\n"
        ranking_text += f"Dominant eigenvalue: {dominant_eigenval:.3f}\n"
        ranking_text += f"(Should be 1.0 for stochastic matrix)\n\n"
        ranking_text += f"Eigenvector = PageRank scores!\n"
        ranking_text += f"This proves PageRank finds the\n"
        ranking_text += f"principal eigenvector of the\n"
        ranking_text += f"Google matrix."
        
        ax2.text(1.02, 1, ranking_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='navy', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ PageRank Visualizer loaded!")
        print("🎯 This shows how Google uses eigenvectors to rank web pages!")
        print(f"📈 Convergence achieved in {len(rank_history)-1} iterations")
        
        return fig
    
    def _create_google_matrix(self, adjacency, damping_factor=0.85):
        """Create Google matrix from adjacency matrix"""
        n = adjacency.shape[0]
        
        # Create transition matrix (column stochastic)
        column_sums = adjacency.sum(axis=0)
        # Handle dangling nodes (pages with no outgoing links)
        column_sums[column_sums == 0] = 1
        transition_matrix = adjacency / column_sums
        
        # Add damping factor (random jump probability)
        google_matrix = damping_factor * transition_matrix + (1 - damping_factor) / n * np.ones((n, n))
        
        return google_matrix

    def create_data_transformation_demo(self):
        """
        Data transformation and PCA preview
        Shows how linear algebra applies to real data analysis
        """
        print("🎨 Starting Data Transformation Demo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('📊 Data Analysis with Linear Algebra', fontsize=16, color='cyan')
        
        # Generate sample data (height vs weight)
        np.random.seed(42)
        n_samples = 100
        
        # Create correlated data
        true_height = np.random.normal(170, 10, n_samples)  # cm
        true_weight = 0.8 * true_height - 80 + np.random.normal(0, 5, n_samples)  # kg
        
        data_original = np.column_stack([true_height, true_weight])
        
        # 1. Original data scatter plot
        ax1.scatter(data_original[:, 0], data_original[:, 1], alpha=0.6, c='lightblue')
        ax1.set_xlabel('Height (cm)')
        ax1.set_ylabel('Weight (kg)')
        ax1.set_title('Original Data', color='lightblue')
        ax1.grid(True, alpha=0.3)
        
        # Calculate and show mean
        mean_point = np.mean(data_original, axis=0)
        ax1.plot(mean_point[0], mean_point[1], 'ro', markersize=10, label='Mean')
        ax1.legend()
        
        # 2. Centered data
        centered_data = data_original - mean_point
        ax2.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.6, c='lightgreen')
        ax2.set_xlabel('Height - Mean (cm)')
        ax2.set_ylabel('Weight - Mean (kg)')
        ax2.set_title('Centered Data', color='lightgreen')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Covariance matrix visualization
        cov_matrix = np.cov(centered_data.T)
        
        # Display covariance matrix as heatmap
        im = ax3.imshow(cov_matrix, cmap='RdBu', aspect='auto')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Height', 'Weight'])
        ax3.set_yticklabels(['Height', 'Weight'])
        ax3.set_title('Covariance Matrix', color='lightyellow')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax3.text(j, i, f'{cov_matrix[i, j]:.1f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Principal components (preview of PCA)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Plot data with principal components
        ax4.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.6, c='lightcoral')
        
        # Draw principal component directions
        scale_factor = 3 * np.sqrt(eigenvals)
        
        # First principal component (direction of maximum variance)
        pc1 = eigenvecs[:, 0] * scale_factor[0]
        ax4.arrow(0, 0, pc1[0], pc1[1], head_width=1, head_length=1,
                 fc='yellow', ec='yellow', linewidth=3, label=f'PC1 (λ={eigenvals[0]:.1f})')
        ax4.arrow(0, 0, -pc1[0], -pc1[1], head_width=1, head_length=1,
                 fc='yellow', ec='yellow', linewidth=3, alpha=0.7)
        
        # Second principal component
        pc2 = eigenvecs[:, 1] * scale_factor[1]
        ax4.arrow(0, 0, pc2[0], pc2[1], head_width=1, head_length=1,
                 fc='orange', ec='orange', linewidth=3, label=f'PC2 (λ={eigenvals[1]:.1f})')
        ax4.arrow(0, 0, -pc2[0], -pc2[1], head_width=1, head_length=1,
                 fc='orange', ec='orange', linewidth=3, alpha=0.7)
        
        ax4.set_xlabel('Height - Mean (cm)')
        ax4.set_ylabel('Weight - Mean (kg)')
        ax4.set_title('Principal Components', color='lightyellow')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axis('equal')
        
        # Add mathematical insight
        insight_text = f'''🔬 Mathematical Insights:

Covariance Matrix:
C = X^T X / (n-1)

Eigenvalue Decomposition:
C = Q Λ Q^T

Principal Components:
- PC1 explains {eigenvals[0]/(eigenvals[0]+eigenvals[1])*100:.1f}% of variance
- PC2 explains {eigenvals[1]/(eigenvals[0]+eigenvals[1])*100:.1f}% of variance

Total Variance: {np.sum(eigenvals):.1f}
Condition Number: {eigenvals[0]/eigenvals[1]:.2f}

🎯 Next Week: We'll use PCA for 
dimensionality reduction!'''
        
        fig.text(0.02, 0.5, insight_text, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Data Transformation Demo loaded!")
        print("🎯 This preview shows how linear algebra enables data science!")
        print("📊 Principal components find the natural axes in data")
        
        return fig

    def run_comprehensive_demo(self):
        """
        Run all visualizations in sequence
        Complete visual learning experience for Week 1
        """
        print("🚀 Starting Comprehensive Linear Algebra Visual Journey!")
        print("=" * 60)
        
        demos = [
            ("Matrix Transformations", self.create_matrix_transformer),
            ("Vector Operations", self.create_vector_playground), 
            ("PageRank Algorithm", self.create_pagerank_visualizer),
            ("Data Analysis", self.create_data_transformation_demo)
        ]
        
        print("📋 Available Demos:")
        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        
        print("\n🎮 Interactive Demo Controls:")
        print("  • Use sliders to change parameters in real-time")
        print("  • Try preset transformation buttons")
        print("  • Check/uncheck operations to see different results")
        print("  • Watch convergence in PageRank algorithm")
        
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
        print("🎉 All visual demos loaded successfully!")
        print("🎯 Key Learning Objectives Achieved:")
        print("  ✅ Matrix transformations and their geometric meaning")
        print("  ✅ Vector operations and their applications")  
        print("  ✅ Eigenvalues/eigenvectors in real algorithms")
        print("  ✅ Linear algebra foundations for data science")
        print("\n🚀 Ready for Week 2: Calculus - The Engine of Learning!")
        
        return figures

# Convenience functions for direct execution
def matrix_transformer():
    """Quick access to matrix transformation visualizer"""
    viz = LinearAlgebraVisualizer()
    return viz.create_matrix_transformer()

def vector_playground():
    """Quick access to vector operations playground"""
    viz = LinearAlgebraVisualizer()
    return viz.create_vector_playground()

def pagerank_demo():
    """Quick access to PageRank visualization"""
    viz = LinearAlgebraVisualizer()
    return viz.create_pagerank_visualizer()

def data_analysis_demo():
    """Quick access to data analysis demo"""
    viz = LinearAlgebraVisualizer()
    return viz.create_data_transformation_demo()

def run_all_demos():
    """Run complete visual learning experience"""
    viz = LinearAlgebraVisualizer()
    return viz.run_comprehensive_demo()

# Main execution for browser environment
if __name__ == "__main__":
    print("🧠 Neural Odyssey - Week 1: Linear Algebra Visualizations")
    print("=" * 60)
    print("🎨 Available Functions:")
    print("  • matrix_transformer() - Interactive matrix transformations")
    print("  • vector_playground() - Vector operations playground")
    print("  • pagerank_demo() - PageRank algorithm visualization")
    print("  • data_analysis_demo() - Data analysis with linear algebra")
    print("  • run_all_demos() - Complete visual learning experience")
    print("\n🚀 Type any function name to start!")
    print("💡 Example: run_all_demos()")
    
    # For automatic execution in browser
    # Uncomment the next line to auto-run all demos
    # run_all_demos()