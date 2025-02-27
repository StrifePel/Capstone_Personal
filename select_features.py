#!/usr/bin/env python3
"""
Command-line interface for selecting features and running the model with different feature combinations.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import torch
import logging
from datetime import datetime

from config import config, get_active_features
from data_utils import create_dataloaders
from model import UNet, FeatureImportanceAnalyzer
from train import Trainer

# All available features
ALL_FEATURES = [
    'NDVI',           # Normalized Difference Vegetation Index 
    'elevation',      # Elevation data
    'population',     # Population density
    'pdsi',           # Palmer Drought Severity Index
    'vs',             # Wind speed
    'pr',             # Precipitation
    'sph',            # Specific humidity
    'tmmx',           # Maximum temperature
    'th',             # Wind direction
    'tmmn',           # Minimum temperature
    'erc',            # Energy Release Component (fire danger)
]

# Feature groups based on domain knowledge
FEATURE_GROUPS = {
    'topographic': ['elevation'],
    'vegetation': ['NDVI'],
    'weather': ['vs', 'pr', 'sph', 'tmmx', 'th', 'tmmn'],
    'fire_danger': ['erc', 'pdsi'],
    'human': ['population'],
    'temperature': ['tmmx', 'tmmn'],
    'moisture': ['pr', 'sph', 'pdsi'],
    'wind': ['vs', 'th'],
}

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def train_with_features(active_features, args, logger):
    """Train a model with the specified active features."""
    if not active_features:
        logger.error("No features selected! Please select at least one feature.")
        return False
    
    # Update config for this run
    run_config = config.copy()
    run_config['active_features'] = active_features
    run_config['n_channels'] = len(active_features)
    
    # Generate a unique run ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feature_hash = "_".join(sorted(active_features))
    run_id = f"{timestamp}_{feature_hash}"
    
    # Set up directories
    checkpoint_dir = Path(config['checkpoint_dir']) / run_id
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders with selected features
    logger.info(f"Creating dataloaders with features: {active_features}")
    train_loader, val_loader = create_dataloaders(
        run_config['train_path'],
        run_config['val_path'],
        run_config['batch_size'],
        run_config['num_workers'],
        run_config['cache_dir'],
        active_features=active_features
    )
    
    # Initialize model
    logger.info(f"Initializing model with {len(active_features)} channels")
    model = UNet(
        n_channels=len(active_features),
        n_classes=run_config['n_classes'],
        feature_names=active_features
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=run_config,
        run_id=run_id
    )
    
    # Train the model
    logger.info(f"Starting training with features: {active_features}")
    trainer.train()
    
    # Save feature configuration
    with open(checkpoint_dir / "feature_config.json", 'w') as f:
        json.dump({
            'active_features': active_features,
            'n_channels': len(active_features)
        }, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {checkpoint_dir}")
    return True

def analyze_feature_combinations(args, logger):
    """Run a sequence of feature selection experiments."""
    if args.mode == 'groups':
        # Train with each feature group
        for group_name, features in FEATURE_GROUPS.items():
            logger.info(f"\n===== Training with feature group: {group_name} =====")
            train_with_features(features, args, logger)
    
    elif args.mode == 'individual':
        # Train with each feature individually
        for feature in ALL_FEATURES:
            logger.info(f"\n===== Training with single feature: {feature} =====")
            train_with_features([feature], args, logger)
    
    elif args.mode == 'incremental':
        # Train with incrementally adding features
        # First, get feature importance if available
        if args.importance_file:
            try:
                with open(args.importance_file, 'r') as f:
                    importance_data = json.load(f)
                # Sort features by importance
                features_ordered = sorted(
                    importance_data.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                ordered_features = [feat for feat, _ in features_ordered]
            except Exception as e:
                logger.error(f"Error loading importance file: {e}")
                ordered_features = ALL_FEATURES
        else:
            ordered_features = ALL_FEATURES
        
        # Train with incrementally adding features
        current_features = []
        for feature in ordered_features:
            current_features.append(feature)
            logger.info(f"\n===== Training with {len(current_features)} features: {current_features} =====")
            train_with_features(current_features, args, logger)
    
    elif args.mode == 'ablation':
        # Train with all features except one (leave-one-out)
        for feature_to_remove in ALL_FEATURES:
            active_features = [f for f in ALL_FEATURES if f != feature_to_remove]
            logger.info(f"\n===== Training without feature: {feature_to_remove} =====")
            train_with_features(active_features, args, logger)
    
    elif args.mode == 'custom':
        # Train with custom feature selection
        if not args.features:
            logger.error("No features specified for custom mode. Use --features to specify features.")
            return
        
        logger.info(f"\n===== Training with custom feature selection: {args.features} =====")
        train_with_features(args.features, args, logger)
    
    else:
        logger.error(f"Unknown mode: {args.mode}")

def main():
    parser = argparse.ArgumentParser(description='Feature selection for wildfire prediction model')
    
    # Main mode selection
    parser.add_argument('--mode', type=str, choices=['groups', 'individual', 'incremental', 'ablation', 'custom'], 
                      default='custom', help='Feature selection mode')
    
    # Custom feature selection
    parser.add_argument('--features', type=str, nargs='+', 
                      help='Specific features to use (for custom mode)')
    
    # Additional options
    parser.add_argument('--importance-file', type=str, 
                      help='JSON file with feature importance scores (for incremental mode)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Display available features and groups
    logger.info("Available features:")
    for feature in ALL_FEATURES:
        logger.info(f"  - {feature}")
    
    logger.info("\nAvailable feature groups:")
    for group, features in FEATURE_GROUPS.items():
        logger.info(f"  - {group}: {features}")
    
    # Run feature selection process
    analyze_feature_combinations(args, logger)

if __name__ == "__main__":
    main()