"""
Regime Comparison Visualization Functions

Helper functions for comparing regime detection methods and visualizing
ensemble performance in Notebook 07.

Functions:
- plot_regime_distributions_comparison: Compare regime frequency distributions
- plot_transition_matrices_comparison: Compare transition matrices across methods
- plot_crisis_validation_comparison: Validate regime detection during crisis periods
- plot_method_agreement_heatmap: Visualize pairwise agreement between methods
- plot_ensemble_transition_analysis: Analyze ensemble transition patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import transition matrix computation from analysis module
from src.utils.analysis.regime_analysis import compute_transition_matrix


def plot_regime_distributions_comparison(garch_labels, hmm_labels, kmeans_labels, save_path=None):
    """
    Create side-by-side bar charts comparing regime distributions across methods.
    
    Args:
        garch_labels: GARCH regime labels (0, 1, 2)
        hmm_labels: HMM regime labels (0, 1, 2)
        kmeans_labels: K-means regime labels (0, 1, 2)
        save_path: Optional path to save figure
    """
    regime_names = ['Low\nVolatility', 'Medium\nVolatility', 'High\nVolatility']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [
        ('GARCH', garch_labels),
        ('HMM', hmm_labels),
        ('K-means', kmeans_labels)
    ]
    
    for ax, (method_name, labels) in zip(axes, methods):
        # Count regime frequencies
        counts = np.array([np.sum(labels == i) for i in range(3)])
        percentages = counts / len(labels) * 100
        
        # Create bar chart
        bars = ax.bar(regime_names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Number of Days', fontsize=11)
        ax.set_title(f'{method_name} Regime Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_transition_matrices_comparison(garch_labels, hmm_labels, kmeans_labels, save_path=None):
    """
    Create side-by-side heatmaps comparing transition matrices across methods.
    
    Args:
        garch_labels: GARCH regime labels (0, 1, 2)
        hmm_labels: HMM regime labels (0, 1, 2)
        kmeans_labels: K-means regime labels (0, 1, 2)
        save_path: Optional path to save figure
    """
    # Compute transition matrices
    garch_trans = compute_transition_matrix(garch_labels)
    hmm_trans = compute_transition_matrix(hmm_labels)
    kmeans_trans = compute_transition_matrix(kmeans_labels)
    
    regime_labels = ['Low Vol', 'Med Vol', 'High Vol']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    matrices = [
        ('GARCH', garch_trans, garch_labels),
        ('HMM', hmm_trans, hmm_labels),
        ('K-means', kmeans_trans, kmeans_labels)
    ]
    
    for ax, (method_name, trans_matrix, labels) in zip(axes, matrices):
        # Create heatmap
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=regime_labels, yticklabels=regime_labels,
                   vmin=0, vmax=1, cbar_kws={'label': 'Probability'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'{method_name} Transition Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Next Regime', fontsize=10)
        ax.set_ylabel('Current Regime', fontsize=10)
        
        # Add persistence metrics below title
        persistence = [trans_matrix[i, i] for i in range(3)]
        avg_persistence = np.mean(persistence)
        
        ax.text(0.5, -0.15, 
               f'Avg Persistence: {avg_persistence:.1%}',
               ha='center', va='top', transform=ax.transAxes,
               fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_crisis_validation_comparison(dates, garch_labels, hmm_labels, kmeans_labels, 
                                     volatility, crisis_periods, save_path=None):
    """
    Compare how each method detects regimes during crisis periods.
    
    Args:
        dates: Array of dates
        garch_labels: GARCH regime labels (0, 1, 2)
        hmm_labels: HMM regime labels (0, 1, 2)
        kmeans_labels: K-means regime labels (0, 1, 2)
        volatility: Market volatility values
        crisis_periods: Dict with crisis names as keys and (start, end) date tuples as values
        save_path: Optional path to save figure
    """
    dates_pd = pd.to_datetime(dates)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    methods = [
        ('GARCH', garch_labels),
        ('HMM', hmm_labels),
        ('K-means', kmeans_labels)
    ]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Low, Medium, High volatility
    
    for ax, (method_name, labels) in zip(axes, methods):
        # Plot volatility as background
        ax.plot(dates_pd, volatility, color='gray', alpha=0.3, linewidth=1, label='Market Volatility')
        
        # Scatter plot colored by regime
        for regime in range(3):
            mask = labels == regime
            regime_names = ['Low Volatility', 'Medium Volatility', 'High Volatility']
            ax.scatter(dates_pd[mask], volatility[mask], 
                      c=colors[regime], s=15, alpha=0.6,
                      label=regime_names[regime])
        
        # Highlight crisis periods
        for crisis_name, (start, end) in crisis_periods.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            ax.axvspan(start_date, end_date, alpha=0.15, color='red', 
                      label=crisis_name if ax == axes[0] else '')
            
            # Add crisis label
            mid_date = start_date + (end_date - start_date) / 2
            ax.text(mid_date, ax.get_ylim()[1] * 0.95, crisis_name,
                   ha='center', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_ylabel('Volatility (%)', fontsize=10)
        ax.set_title(f'{method_name} Crisis Detection', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    axes[-1].set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_method_agreement_heatmap(garch_labels, hmm_labels, kmeans_labels, save_path=None):
    """
    Visualize pairwise agreement rates between methods.
    
    Args:
        garch_labels: GARCH regime labels (0, 1, 2)
        hmm_labels: HMM regime labels (0, 1, 2)
        kmeans_labels: K-means regime labels (0, 1, 2)
        save_path: Optional path to save figure
    """
    # Compute pairwise agreement matrix
    methods = ['GARCH', 'HMM', 'K-means']
    labels_list = [garch_labels, hmm_labels, kmeans_labels]
    
    agreement_matrix = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            agreement_matrix[i, j] = np.mean(labels_list[i] == labels_list[j])
    
    # Compute 3-way agreement
    three_way_agreement = np.mean(
        (garch_labels == hmm_labels) & (hmm_labels == kmeans_labels)
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of pairwise agreement
    sns.heatmap(agreement_matrix, annot=True, fmt='.1%', cmap='RdYlGn',
               xticklabels=methods, yticklabels=methods,
               vmin=0, vmax=1, cbar_kws={'label': 'Agreement Rate'},
               ax=ax1, linewidths=2, linecolor='white', square=True)
    
    ax1.set_title('Pairwise Method Agreement', fontsize=13, fontweight='bold')
    
    # Bar chart of agreement statistics
    pairwise_agreements = [
        agreement_matrix[0, 1],  # GARCH-HMM
        agreement_matrix[0, 2],  # GARCH-K-means
        agreement_matrix[1, 2],  # HMM-K-means
        three_way_agreement
    ]
    
    labels = ['GARCH\nvs\nHMM', 'GARCH\nvs\nK-means', 'HMM\nvs\nK-means', 'All Three\nAgree']
    colors_bar = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    bars = ax2.bar(labels, pairwise_agreements, color=colors_bar, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for bar, val in zip(bars, pairwise_agreements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Agreement Rate', fontsize=11)
    ax2.set_title('Method Agreement Statistics', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Return statistics for table
    return {
        'garch_hmm': agreement_matrix[0, 1],
        'garch_kmeans': agreement_matrix[0, 2],
        'hmm_kmeans': agreement_matrix[1, 2],
        'three_way': three_way_agreement
    }


def plot_ensemble_transition_analysis(ensemble_labels, method_name='Ensemble', save_path=None):
    """
    Analyze and visualize ensemble transition patterns and persistence.
    
    Args:
        ensemble_labels: Ensemble regime labels (0, 1, 2)
        method_name: Name of ensemble method for title
        save_path: Optional path to save figure
    """
    # Compute transition matrix
    trans_matrix = compute_transition_matrix(ensemble_labels)
    
    regime_labels = ['Low Vol', 'Med Vol', 'High Vol']
    regime_names = ['Low Volatility', 'Medium Volatility', 'High Volatility']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Transition matrix heatmap
    sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=regime_labels, yticklabels=regime_labels,
               vmin=0, vmax=1, cbar_kws={'label': 'Transition Probability'},
               ax=ax1, linewidths=1, linecolor='gray', square=True)
    
    ax1.set_title(f'{method_name} Transition Matrix', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Next Regime', fontsize=11)
    ax1.set_ylabel('Current Regime', fontsize=11)
    
    # Panel 2: Persistence and duration analysis
    persistence = [trans_matrix[i, i] for i in range(3)]
    durations = [1 / (1 - p) if p < 1.0 else np.inf for p in persistence]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    x = np.arange(3)
    width = 0.35
    
    # Persistence bars
    bars1 = ax2.bar(x - width/2, persistence, width, label='Persistence Probability',
                   color=colors, alpha=0.7, edgecolor='black')
    
    # Duration bars (normalized to 0-1 scale for visualization)
    max_duration = max([d for d in durations if d != np.inf])
    durations_normalized = [d / max_duration if d != np.inf else 1.0 for d in durations]
    bars2 = ax2.bar(x + width/2, durations_normalized, width, label='Avg Duration (norm)',
                   color=colors, alpha=0.4, edgecolor='black')
    
    # Add value labels
    for i, (bar, pers, dur) in enumerate(zip(bars1, persistence, durations)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pers:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add duration text
        if dur != np.inf:
            ax2.text(x[i] + width/2, durations_normalized[i] + 0.02,
                    f'{dur:.1f}d',
                    ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Probability / Normalized Duration', fontsize=11)
    ax2.set_title(f'{method_name} Persistence & Duration', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_names, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Return statistics
    return {
        'transition_matrix': trans_matrix,
        'persistence': persistence,
        'durations': durations
    }

