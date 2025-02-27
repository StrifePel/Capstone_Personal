#!/usr/bin/env python3
"""
Script to analyze feature importance using various methods:
1. Gradient-based feature importance
2. Comparative analysis of model performance with different feature sets
3. Visualizations of feature impact
"""

import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader

from config import config, get_active_features
from data_utils import WildfireDataset
from model import UNet, FeatureImportanceAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model_and_data(checkpoint_path, device, active_features=None):
    """Load model from checkpoint and prepare data loader."""
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Determine active features
    if active_features is None:
        if 'active_features' in checkpoint:
            active_features = checkpoint['active_features']
            logger.info(f"Using features from checkpoint: {active_features}")
        else:
            active_features = get_active_features()
            logger.info(f"Using features from config: {active_features}")
    
    # Initialize model
    model = UNet(
        n_channels=len(active_features),
        n_classes=1,
        feature_names=active_features
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info("Model initialized successfully")
    
    # Create validation dataset
    val_dataset = WildfireDataset(
        tfrecord_path=config['val_path'],
        cache_dir=config['cache_dir'],
        active_features=active_features
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # Smaller batch size for detailed analysis
        shuffle=False,
        num_workers=2
    )
    logger.info(f"Created validation dataloader with {len(val_dataset)} samples")
    
    return model, val_loader, active_features

def analyze_gradients(model, val_loader, active_features, device, save_dir):
    """Perform gradient-based feature importance analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing feature importance using gradient magnitudes...")
    
    analyzer = FeatureImportanceAnalyzer(model, active_features)
    importance = analyzer.analyze_gradient_based_importance(val_loader, device)
    
    # Create importance dataframe
    importance_df = pd.DataFrame(importance, columns=['feature', 'importance'])
    
    # Log results
    logger.info("Feature importance ranking:")
    for feature, score in importance:
        logger.info(f"  {feature}: {score:.6f}")
    
    # Save to file
    with open(save_dir / "feature_importance_gradient.json", 'w') as f:
        json.dump({feat: float(score) for feat, score in importance}, f, indent=2)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='feature', y='importance', data=importance_df)
    plt.title('Feature Importance Based on Gradient Magnitudes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance_gradient.png", dpi=300)
    plt.close()
    
    return importance_df

def collect_experiment_results(experiments_dir):
    """Collect and compare results from various feature selection experiments."""
    logger = logging.getLogger(__name__)
    logger.info(f"Collecting results from experiment directory: {experiments_dir}")
    
    results = []
    
    # Find all feature experiment directories
    experiment_dirs = [d for d in Path(experiments_dir).iterdir() if d.is_dir()]
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    for exp_dir in experiment_dirs:
        # Look for metrics.json and feature configuration
        metrics_file = exp_dir / "metrics.json"
        feature_file = exp_dir / "feature_config.json"
        
        if not metrics_file.exists():
            logger.warning(f"No metrics.json found in {exp_dir}")
            continue
            
        if not feature_file.exists():
            logger.warning(f"No feature_config.json found in {exp_dir}")
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Load feature configuration
        with open(feature_file, 'r') as f:
            feature_config = json.load(f)
        
        # Extract metrics for threshold 0.5 (or first threshold if not present)
        threshold = "0.5" if "0.5" in metrics else list(metrics.keys())[0]
        threshold_metrics = metrics[threshold]
        
        # Combine information
        result = {
            'experiment': exp_dir.name,
            'features': feature_config.get('active_features', []),
            'num_features': feature_config.get('n_channels', 0),
            'f1_score': threshold_metrics.get('f1_score', 0),
            'precision': threshold_metrics.get('precision', 0),
            'recall': threshold_metrics.get('recall', 0),
            'auc_roc': threshold_metrics.get('auc_roc', 0)
        }
        
        results.append(result)
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        logger.info(f"Collected results for {len(results_df)} experiments")
        return results_df
    else:
        logger.warning("No valid experiment results found")
        return None

def visualize_comparative_results(results_df, save_dir):
    """Visualize comparative results from different feature combinations."""
    logger = logging.getLogger(__name__)
    if results_df is None or len(results_df) == 0:
        logger.warning("No results to visualize")
        return
    
    # 1. Number of features vs. performance metrics
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['num_features'], results_df['f1_score'], 'o-', label='F1 Score')
    plt.plot(results_df['num_features'], results_df['auc_roc'], 's-', label='AUC ROC')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Number of Features')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / "performance_vs_features.png", dpi=300)
    plt.close()
    
    # 2. Feature presence heatmap
    # Create a matrix of feature presence
    all_features = set()
    for features in results_df['features']:
        all_features.update(features)
    
    all_features = sorted(list(all_features))
    feature_matrix = np.zeros((len(results_df), len(all_features)))
    
    for i, features in enumerate(results_df['features']):
        for j, feature in enumerate(all_features):
            if feature in features:
                feature_matrix[i, j] = 1
    
    # Create a new DataFrame with feature presence and F1 score
    heatmap_df = pd.DataFrame(feature_matrix, columns=all_features)
    heatmap_df['F1 Score'] = results_df['f1_score']
    
    # Sort by F1 score
    heatmap_df = heatmap_df.sort_values('F1 Score', ascending=False)
    f1_scores = heatmap_df['F1 Score']
    heatmap_df = heatmap_df.drop('F1 Score', axis=1)
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_df, cmap='Blues', cbar=False)
    
    # Add F1 scores as text
    ax = plt.gca()
    ax.set_title('Feature Combinations Sorted by Performance')
    ax.set_xlabel('Features')
    ax.set_ylabel('Experiments')
    
    # Add F1 scores as a separate column
    for i, score in enumerate(f1_scores):
        plt.text(len(all_features) + 0.5, i + 0.5, f"{score:.4f}", 
                 ha='center', va='center', fontweight='bold')
    plt.text(len(all_features) + 0.5, -0.3, 'F1 Score',
             ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "feature_combinations_heatmap.png", dpi=300)
    plt.close()
    
    # 3. Top single feature performance
    if len(results_df[results_df['num_features'] == 1]) > 0:
        single_feature_df = results_df[results_df['num_features'] == 1].copy()
        single_feature_df['feature'] = single_feature_df['features'].apply(lambda x: x[0] if x else "Unknown")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='feature', y='f1_score', data=single_feature_df)
        plt.title('Performance of Individual Features')
        plt.xlabel('Feature')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / "single_feature_performance.png", dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {save_dir}")
    
    # 4. Feature occurrence in top-performing models
    # Consider top 25% of models by F1 score
    top_quartile = np.percentile(results_df['f1_score'], 75)
    top_models = results_df[results_df['f1_score'] >= top_quartile]
    
    feature_counts = {}
    for features in top_models['features']:
        for feature in features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
    
    if feature_counts:
        feature_freq_df = pd.DataFrame({
            'feature': list(feature_counts.keys()),
            'frequency': list(feature_counts.values())
        })
        feature_freq_df['frequency_pct'] = 100 * feature_freq_df['frequency'] / len(top_models)
        feature_freq_df = feature_freq_df.sort_values('frequency', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='feature', y='frequency_pct', data=feature_freq_df)
        plt.title('Feature Frequency in Top Performing Models')
        plt.xlabel('Feature')
        plt.ylabel('Frequency (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / "top_model_feature_frequency.png", dpi=300)
        plt.close()
        
        # Save feature frequency data
        feature_freq_df.to_csv(save_dir / "top_model_feature_frequency.csv", index=False)
        logger.info(f"Feature frequency analysis saved to {save_dir}")
        
        return feature_freq_df