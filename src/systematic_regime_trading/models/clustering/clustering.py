"""
Helpers for k-means clustering to identify volatility regimes.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

 



def clustering_pipeline(X, pca_vision=False, train=True, prediction_only=False, pipe=None):
    """
    Build or use a simple pipeline (scaler + PCA + k-means) and return labels.

    Args:
        X: Feature matrix
        pca_vision: If True, plot PCA scatter with clusters
        train: If True, fit the provided pipeline
        prediction_only: If True, only predict with provided pipeline
        pipe: Optional prebuilt pipeline

    Returns:
        (labels, fitted_pipeline)
    """
    # Check for and handle NaN values
    if isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValueError(f"Input contains {np.isnan(X).sum()} NaN values. Please clean data before clustering.")
    else:  # DataFrame
        if X.isnull().any().any():
            nan_counts = X.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            raise ValueError(f"Input contains NaN values in columns: {nan_cols.to_dict()}. Please clean data before clustering.")
    
    if pipe is not None and prediction_only:
        # Use provided pipeline to predict
        labels = pipe.predict(X)

    elif pipe is not None and train:
        # Fit provided pipeline and predict
        labels = pipe.fit_predict(X)
        
    else :
        # Determine number of PCA components based on number of features
        n_features = X.shape[1]
        n_components = min(2, n_features)  # Use at most 2 components, but not more than available features
        
        if n_features < 3:
            # Not enough features for meaningful clustering with PCA
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(n_clusters=3, random_state=42))
            ])
        else:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('kmeans', KMeans(n_clusters=3, random_state=42))
            ])
        labels = pipe.fit_predict(X)

        

    # Access the fitted KMeans model
    kmeans_model = pipe.named_steps['kmeans']

    


    if pca_vision == True:

        # Transform data to PCA space for visualization
        X_pca = pipe[:-1].transform(X)
        centroids = kmeans_model.cluster_centers_

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, 
                    edgecolors='black', linewidths=2, label='Centroids')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return labels, pipe


# What you can do next:
""""
kmeans_model = pipe.named_steps['kmeans']

print(f"Converged in {kmeans_model.n_iter_} iterations")
print(f"Inertia: {kmeans_model.inertia_:.2f}")
print(f"Cluster sizes: {np.bincount(labels)}")
centroids = kmeans_model.cluster_centers_
"""

def get_features(file_path):
    import pandas as pd
    # Load features from CSV
    features = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return features

def get_feature_importance(pipe, feature_names):
    """
    Calculate feature importance based on PCA component weights.
    
    Args:
        pipe: Fitted pipeline containing PCA step
        feature_names: List of original feature names
    
    Returns:
        list: Feature names sorted by importance (descending)
    """
    pca = pipe.named_steps['pca']
    
    # Get absolute values of PCA components and sum across components
    # This gives us the total contribution of each feature
    importance = np.abs(pca.components_).sum(axis=0)
    
    # Normalize to get relative importance
    importance = importance / importance.sum()
    
    # Create list of (feature_name, importance) tuples and sort
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted feature names
    return [(name,significance) for name, significance in feature_importance]


def plot_top_features(pipe, feature_names, top_n=5):
    """
    Plot the top N most important features by significance.
    
    Args:
        pipe: Fitted pipeline containing PCA step
        feature_names: List of original feature names
        top_n: Number of top features to display (default: 5, max: 5)
    """
    pca = pipe.named_steps['pca']
    
    # Calculate feature importance
    importance = np.abs(pca.components_).sum(axis=0)
    importance = importance / importance.sum()
    
    # Get top N features (at most 5)
    top_n = min(top_n, 5, len(feature_names))
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(top_n), top_importance, color='steelblue', alpha=0.8)
    plt.xticks(range(top_n), top_features, rotation=45, ha='right')
    plt.ylabel('Relative Importance')
    plt.xlabel('Features')
    plt.title(f'Top {top_n} Most Important Features for Clustering')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()



def plot_regimes(df_feature, labels, x_feature='market_volatility', y_feature='market_direction', 
                 title='K-Means Clustering', figsize=(12, 8), alpha=0.6, s=50, cmap='viridis', labels_name=None):
    """
    Plot clustering results on a 2D scatter plot.
    
    Args:
        df_feature: DataFrame containing the features
        labels: Cluster labels for each data point
        x_feature: Feature name for x-axis (default: 'market_volatility')
        y_feature: Feature name for y-axis (default: 'market_direction')
        title: Plot title (default: 'K-Means Clustering')
        figsize: Figure size tuple (default: (12, 8))
        alpha: Transparency of scatter points (default: 0.6)
        s: Size of scatter points (default: 50)
        cmap: Colormap name (default: 'viridis')
        labels_name: Custom label names dict or 0/1 for preset names (default: None)
    """
    # Set default label names and colors
    if labels_name is None:
        label_names = {0: 'Calm', 1: 'Intense', 2: 'Turbulent'}
    elif labels_name == 1:
        label_names = {0: 'Calm', 1: 'Moderate', 2: 'Turbulent'}
    else:
        label_names = labels_name
    
    # Define color mapping: green -> yellow/orange -> red
    colors = {0: 'green', 1: 'orange', 2: 'red'}  # green, orange, red
    
    plt.figure(figsize=figsize)
    
    # Plot each cluster separately to assign custom colors and labels
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        plt.scatter(
            df_feature[x_feature][mask], 
            df_feature[y_feature][mask],
            c=colors[cluster_id],
            label=label_names.get(cluster_id, f'Cluster {cluster_id}'),
            alpha=alpha,
            s=s
        )
    
    plt.xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{title}: {y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_regimes_timeline(df_feature, labels, feature='market_volatility', 
                          title='Market Volatility Over Time - Colored by Cluster',
                          figsize=(15, 6), alpha=0.6, s=30, cmap='viridis', labels_name=None):
    """
    Plot clustering results over time as a timeline.
    
    Args:
        df_feature: DataFrame containing the features with DatetimeIndex
        labels: Cluster labels for each data point
        feature: Feature name to plot on y-axis (default: 'market_volatility')
        title: Plot title (default: 'Market Volatility Over Time - Colored by Cluster')
        figsize: Figure size tuple (default: (15, 6))
        alpha: Transparency of scatter points (default: 0.6)
        s: Size of scatter points (default: 30)
        cmap: Colormap name (default: 'viridis')
        labels_name: Custom label names dict or 0/1 for preset names (default: None)
    """
    # Set default label names
    if labels_name is None:
        label_names = {0: 'Calm', 1: 'Intense', 2: 'Turbulent'}
    elif labels_name == 1:
        label_names = {0: 'Calm', 1: 'Moderate', 2: 'Turbulent'}
    else:
        label_names = labels_name
    
    # Define color mapping: green -> yellow/orange -> red
    colors = {0: 'green', 1: 'orange', 2: 'red'}  # green, orange, red
    
    plt.figure(figsize=figsize)
    
    # Plot each cluster separately to assign custom colors and labels
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        plt.scatter(
            df_feature.index[mask], 
            df_feature[feature][mask],
            c=colors[cluster_id],
            label=label_names.get(cluster_id, f'Cluster {cluster_id}'),
            alpha=alpha,
            s=s
        )
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def ensemble_voting(labels_model1, labels_model2, weights=(0.5, 0.5)):
    """
    Combine two model predictions using weighted voting with intermediate classes.
    
    Parameters:
    -----------
    labels_model1 : array
        Predictions from first model (VolVol)
    labels_model2 : array
        Predictions from second model (MoodIndex)
    weights : tuple
        Voting weights (default 50/50)
        
    Returns:
    --------
    array : Ensemble predictions with intermediate classes (0, 0.5, 1, 1.5, 2)
    
    Examples:
    ---------
    - Both predict 0 → 0 (Calm)
    - VolVol=0, MoodIndex=1 → 0.5 (Calm-Moderate transition)
    - Both predict 1 → 1 (Moderate/Intense)
    - VolVol=1, MoodIndex=2 → 1.5 (Moderate-Turbulent transition)
    - Both predict 2 → 2 (Turbulent)
    """
    ensemble_labels = []
    
    for label1, label2 in zip(labels_model1, labels_model2):
        if label1 == label2:
            # Models agree - use the agreed label
            ensemble_labels.append(float(label1))
        else:
            # Models disagree - use average as intermediate class
            intermediate_label = (label1 + label2) / 2.0
            ensemble_labels.append(intermediate_label)
    
    return np.array(ensemble_labels)

def plot_transition_matrix(labels, regime_names, title):
    """Create and plot regime transition probability matrix."""
    transitions = pd.DataFrame({
        'from_regime': labels[:-1],
        'to_regime': labels[1:]
    })
    
    transition_matrix = pd.crosstab(
        transitions['from_regime'],
        transitions['to_regime'],
        normalize='index'
    )
    
    # Rename with regime names
    transition_matrix.index = transition_matrix.index.map(regime_names)
    transition_matrix.columns = transition_matrix.columns.map(regime_names)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        cmap='Reds', 
        fmt='.2f',
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Transition Probability'}
    )
    plt.title(title)
    plt.xlabel('To Regime')
    plt.ylabel('From Regime')
    plt.tight_layout()
    plt.show()
    
    return transition_matrix

# Rearrange cluster labels based on mean volatility
def rearrange_labels_by_feature(df, labels, feature_name):
    """Reorder cluster labels by ascending mean of target feature."""
    df_temp = df.copy()
    df_temp['cluster'] = labels
    cluster_means = df_temp.groupby('cluster')[feature_name].mean().sort_values()
    label_mapping = {old: new for new, old in enumerate(cluster_means.index)}
    return np.array([label_mapping[label] for label in labels])


def plot_regimes_timeline_5(df_feature, labels, feature='market_volatility', 
                          title='Market Volatility Over Time - Colored by Cluster',
                          figsize=(15, 6), alpha=0.6, s=30, cmap='viridis', labels_name=None):
    """
    Plot clustering results over time as a timeline.
    Supports intermediate classes (0.5, 1.5) from ensemble voting.
    """
    # Set default label names including intermediate classes
    if labels_name is None:
        label_names = {
            0: 'Calm', 
            0.5: 'Calm-Moderate',
            1: 'Intense', 
            1.5: 'Moderate-Turbulent',
            2: 'Turbulent'
        }
    elif labels_name == 1:
        label_names = {
            0: 'Calm',
            0.5: 'Calm-Moderate',
            1: 'Moderate',
            1.5: 'Moderate-Turbulent',
            2: 'Turbulent'
        }
    else:
        label_names = labels_name
    
    # Define color mapping with intermediate colors
    colors = {
        0: '#2ecc71',      # green (Calm)
        0.5: "#ccff00",    # yellow (Calm-Moderate)
        1: '#f1c40f',      # orange (Moderate/Intense)
        1.5: '#f39c12',    # dark orange (Moderate-Turbulent)
        2: '#e74c3c'       # red (Turbulent)
    }
    
    plt.figure(figsize=figsize)
    
    # Plot each cluster separately to assign custom colors and labels
    unique_labels = sorted(np.unique(labels))
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        plt.scatter(
            df_feature.index[mask], 
            df_feature[feature][mask],
            c=colors.get(cluster_id, 'gray'),
            label=label_names.get(cluster_id, f'Cluster {cluster_id}'),
            alpha=alpha,
            s=s
        )
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Compute transition matrix
def compute_transition_matrix(regimes):
    """Compute transition probability matrix from regime sequence."""
    n_states = len(np.unique(regimes))
    transition_counts = np.zeros((n_states, n_states))
    
    # Convert to numpy array for easier indexing
    regime_array = np.array(regimes)
    
    for i in range(len(regime_array) - 1):
        current_state = int(regime_array[i])
        next_state = int(regime_array[i + 1])
        transition_counts[current_state, next_state] += 1
    
    # Convert counts to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_counts, row_sums, 
                                 where=row_sums != 0, out=np.zeros_like(transition_counts))
    
    return transition_probs

def plot_regimes_5(df_feature, labels, x_feature='market_volatility', y_feature='market_direction', 
                   title='K-Means Clustering', figsize=(12, 8), alpha=0.6, s=50, cmap='viridis', labels_name=None):
    """
    Plot clustering results on a 2D scatter plot with support for 5 classes (including intermediate).
    
    Args:
        df_feature: DataFrame containing the features
        labels: Cluster labels for each data point (can include 0.5, 1.5)
        x_feature: Feature name for x-axis (default: 'market_volatility')
        y_feature: Feature name for y-axis (default: 'market_direction')
        title: Plot title (default: 'K-Means Clustering')
        figsize: Figure size tuple (default: (12, 8))
        alpha: Transparency of scatter points (default: 0.6)
        s: Size of scatter points (default: 50)
        cmap: Colormap name (default: 'viridis')
        labels_name: Custom label names dict or 0/1 for preset names (default: None)
    """
    # Set default label names including intermediate classes
    if labels_name is None:
        label_names = {
            0: 'Calm', 
            0.5: 'Calm-Moderate',
            1: 'Intense', 
            1.5: 'Moderate-Turbulent',
            2: 'Turbulent'
        }
    elif labels_name == 1:
        label_names = {
            0: 'Calm',
            0.5: 'Calm-Moderate',
            1: 'Moderate',
            1.5: 'Moderate-Turbulent',
            2: 'Turbulent'
        }
    else:
        label_names = labels_name
    
    # Define color mapping with intermediate colors
    colors = {
        0: '#2ecc71',      # green (Calm)
        0.5: "#ccff00",    # yellow (Calm-Moderate)
        1: '#f1c40f',      # orange (Moderate/Intense)
        1.5: '#f39c12',    # dark orange (Moderate-Turbulent)
        2: '#e74c3c'       # red (Turbulent)
    }
    
    plt.figure(figsize=figsize)
    
    # Plot each cluster separately to assign custom colors and labels
    unique_labels = sorted(np.unique(labels))
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        plt.scatter(
            df_feature[x_feature][mask], 
            df_feature[y_feature][mask],
            c=colors.get(cluster_id, 'gray'),
            label=label_names.get(cluster_id, f'Cluster {cluster_id}'),
            alpha=alpha,
            s=s
        )
    
    plt.xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{title}: {y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_transition_matrix_5(labels, regime_names, title):
    """
    Create and plot regime transition probability matrix for 5 classes (including intermediate).
    
    Args:
        labels: Array of regime labels (can include 0.5, 1.5)
        regime_names: Dictionary mapping label values to regime names
        title: Plot title
        
    Returns:
        DataFrame: Transition probability matrix
    """
    transitions = pd.DataFrame({
        'from_regime': labels[:-1],
        'to_regime': labels[1:]
    })
    
    transition_matrix = pd.crosstab(
        transitions['from_regime'],
        transitions['to_regime'],
        normalize='index'
    )
    
    # Rename with regime names
    transition_matrix.index = transition_matrix.index.map(regime_names)
    transition_matrix.columns = transition_matrix.columns.map(regime_names)
    
    # Ensure consistent ordering for all 5 classes
    all_regimes = [regime_names.get(x, f'{x}') for x in sorted(regime_names.keys())]
    transition_matrix = transition_matrix.reindex(index=all_regimes, columns=all_regimes, fill_value=0)
    
    # Plot heatmap with larger size for 5x5 matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        cmap='Reds', 
        fmt='.2f',
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Transition Probability'},
        square=True
    )
    plt.title(title, fontsize=14)
    plt.xlabel('To Regime', fontsize=12)
    plt.ylabel('From Regime', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return transition_matrix



def remap_ensemble_to_3_states(labels_ensemble, remap_dict=None):
    """
    Remap 5-state ensemble labels (0, 0.5, 1, 1.5, 2) to 3-state labels (0, 1, 2).
    
    Default remapping based on transition probability analysis:
    - 0 (Calm) → 0 (Calm)
    - 0.5 (Calm-Moderate) → 0 (Calm)
    - 1 (Moderate) → 1 (Moderate)
    - 1.5 (Moderate-Turbulent) → 2 (Turbulent)
    - 2 (Turbulent) → 2 (Turbulent)
    
    Args:
        labels_ensemble: Array of 5-state ensemble labels
        remap_dict: Optional custom remapping dictionary. If None, uses default mapping.
        
    Returns:
        np.array: Remapped 3-state labels
    """
    if remap_dict is None:
        remap_dict = {
            0: 0,      # Calm stays Calm
            0.5: 0,    # Calm-Moderate → Calm
            1: 1,      # Moderate stays Moderate
            1.5: 2,    # Moderate-Turbulent → Turbulent
            2: 2       # Turbulent stays Turbulent
        }
    
    # Remap labels
    labels_remapped = np.array([remap_dict[label] for label in labels_ensemble])
    
    return labels_remapped


def analyze_ensemble_remapping(labels_ensemble_5, labels_ensemble_3, labels_volvol, labels_moodindex,
                               regime_names_5=None, regime_names_3=None, remap_dict=None):
    """
    Analyze and display statistics for ensemble regime remapping from 5 to 3 states.
    
    Args:
        labels_ensemble_5: Original 5-state ensemble labels
        labels_ensemble_3: Remapped 3-state ensemble labels
        labels_volvol: VolVol model labels for comparison
        labels_moodindex: MoodIndex model labels for comparison
        regime_names_5: Dictionary mapping 5-state labels to names
        regime_names_3: Dictionary mapping 3-state labels to names
        remap_dict: Remapping dictionary used
        
    Returns:
        dict: Statistics dictionary containing regime distributions and agreement rates
    """
    # Default regime names
    if regime_names_5 is None:
        regime_names_5 = {
            0: 'Calm',
            0.5: 'Calm-Moderate',
            1: 'Moderate',
            1.5: 'Moderate-Turbulent',
            2: 'Turbulent'
        }
    
    if regime_names_3 is None:
        regime_names_3 = {0: 'Calm', 1: 'Moderate', 2: 'Turbulent'}
    
    if remap_dict is None:
        remap_dict = {
            0: 0,
            0.5: 0,
            1: 1,
            1.5: 2,
            2: 2
        }
    
    print("=== Ensemble Regime Remapping: 5 States -> 3 States ===\n")
    print("Remapping Dictionary:")
    for k, v in remap_dict.items():
        original_name = regime_names_5.get(k, str(k))
        new_name = regime_names_3.get(v, str(v))
        print(f"  {k} ({original_name}) -> {v} ({new_name})")
    
    print("\n" + "="*60 + "\n")
    
    # Compare regime distributions
    print("Regime Distribution Comparison:")
    print("\n5-State Ensemble:")
    regime_counts_5 = pd.Series(labels_ensemble_5).value_counts().sort_index()
    for regime in sorted(regime_counts_5.index):
        count = regime_counts_5[regime]
        pct = count / len(labels_ensemble_5) * 100
        regime_name = regime_names_5.get(regime, str(regime))
        print(f"  {regime} ({regime_name}): {count} days ({pct:.1f}%)")
    
    print("\n3-State Ensemble (Remapped):")
    regime_counts_3 = pd.Series(labels_ensemble_3).value_counts().sort_index()
    for regime in sorted(regime_counts_3.index):
        count = regime_counts_3[regime]
        pct = count / len(labels_ensemble_3) * 100
        regime_name = regime_names_3.get(regime, str(regime))
        print(f"  {regime} ({regime_name}): {count} days ({pct:.1f}%)")
    
    print("\n" + "="*60 + "\n")
    
    # Calculate agreement with individual models
    agreement_volvol = np.mean(labels_ensemble_3 == labels_volvol)
    agreement_moodindex = np.mean(labels_ensemble_3 == labels_moodindex)
    
    print("Agreement with Individual Models:")
    print(f"  3-State Ensemble vs VolVol: {agreement_volvol:.1%}")
    print(f"  3-State Ensemble vs MoodIndex: {agreement_moodindex:.1%}")
    
    # Return statistics
    stats = {
        'regime_counts_5': regime_counts_5,
        'regime_counts_3': regime_counts_3,
        'agreement_volvol': agreement_volvol,
        'agreement_moodindex': agreement_moodindex
    }
    
    return stats


# Compute average duration for each regime
def compute_avg_duration(labels, regimes):
    """Compute average consecutive duration for each regime."""
    durations = {r: [] for r in regimes}
    labels_array = np.array(labels)
    current_regime = labels_array[0]
    current_duration = 1
    
    for i in range(1, len(labels_array)):
        if labels_array[i] == current_regime:
            current_duration += 1
        else:
            durations[current_regime].append(current_duration)
            current_regime = labels_array[i]
            current_duration = 1
    # Add last regime duration
    durations[current_regime].append(current_duration)
    
    avg_durations = {r: np.mean(d) if d else 0 for r, d in durations.items()}
    return avg_durations
