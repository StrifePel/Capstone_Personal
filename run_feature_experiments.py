#!/usr/bin/env python3
"""
Script to run a series of experiments with different feature combinations
and analyze the results to determine optimal feature sets.
"""

import argparse
import subprocess
import os
import logging
import time
from pathlib import Path
import json
import itertools
import datetime

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

def setup_logging(log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def run_experiment(feature_set, experiment_dir, logger):
    """Run a single experiment with the specified feature set."""
    feature_str = ' '.join(feature_set)
    feature_hash = '_'.join(sorted(feature_set))
    experiment_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{feature_hash}"
    
    logger.info(f"Running experiment with features: {feature_str}")
    
    # Build command to run training
    cmd = [
        "python", "train.py",
        "--features", *feature_set,
        "--output-dir", str(experiment_dir / experiment_name),
        "--epochs", "50",  # Use fewer epochs for experiments
        "--early-stopping", "10"
    ]
    
    try:
        # Run the command
        logger.info(f"Command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
                
        # Get return code
        return_code = process.poll()
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            logger.info(stdout.strip())
        if stderr:
            logger.error(stderr.strip())
            
        if return_code != 0:
            logger.error(f"Experiment failed with return code {return_code}")
            return False
            
        logger.info(f"Experiment completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        return False

def generate_feature_combinations(mode, min_features=1, max_features=None):
    """Generate feature combinations based on the specified mode."""
    if max_features is None:
        max_features = len(ALL_FEATURES)
        
    if mode == 'all':
        # Generate all possible combinations
        combinations = []
        for r in range(min_features, max_features + 1):
            combinations.extend(list(itertools.combinations(ALL_FEATURES, r)))
        return combinations
    
    elif mode == 'incremental':
        # Start with each feature individually, then add one at a time
        combinations = []
        for base_feature in ALL_FEATURES:
            # Start with the base feature
            current_set = [base_feature]
            combinations.append(tuple(current_set))
            
            # Add additional features one at a time
            remaining = [f for f in ALL_FEATURES if f != base_feature]
            for feature in remaining:
                if len(current_set) < max_features:
                    current_set.append(feature)
                    combinations.append(tuple(current_set))
        return combinations
    
    elif mode == 'random':
        # Generate random combinations
        import random
        random.seed(42)  # For reproducibility
        
        combinations = []
        # Individual features
        for feature in ALL_FEATURES:
            combinations.append((feature,))
            
        # Generate 20 random combinations of varying sizes
        for _ in range(20):
            size = random.randint(min_features, max_features)
            combination = random.sample(ALL_FEATURES, size)
            combinations.append(tuple(combination))
            
        return combinations

def main():
    parser = argparse.ArgumentParser(description='Run multiple feature selection experiments')
    
    parser.add_argument('--mode', type=str, choices=['all', 'incremental', 'random'], 
                      default='random', help='Feature combination generation mode')
    
    parser.add_argument('--output-dir', type=str, default='./feature_experiments',
                      help='Directory to save experiment results')
    
    parser.add_argument('--min-features', type=int, default=1,
                      help='Minimum number of features to use')
    
    parser.add_argument('--max-features', type=int, default=None,
                      help='Maximum number of features to use')
    
    parser.add_argument('--analyze', action='store_true',
                      help='Run analysis after experiments complete')
    
    args = parser.parse_args()
    
    # Setup experiment directory
    experiment_dir = Path(args.output_dir)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    log_file = experiment_dir / "experiments.log"
    logger = setup_logging(log_file)
    
    # Generate feature combinations
    logger.info(f"Generating feature combinations using mode: {args.mode}")
    combinations = generate_feature_combinations(
        args.mode, 
        min_features=args.min_features, 
        max_features=args.max_features
    )
    logger.info(f"Generated {len(combinations)} feature combinations")
    
    # Save combinations
    with open(experiment_dir / "feature_combinations.json", 'w') as f:
        json.dump([list(c) for c in combinations], f, indent=2)
    
    # Run experiments
    successful = 0
    total = len(combinations)
    
    for i, feature_set in enumerate(combinations):
        logger.info(f"\n===== Running experiment {i+1}/{total} =====")
        if run_experiment(feature_set, experiment_dir, logger):
            successful += 1
        
        # Small delay between experiments
        time.sleep(1)
    
    logger.info(f"\n===== All experiments completed =====")
    logger.info(f"Successful: {successful}/{total}")
    
    # Run analysis if requested
    if args.analyze and successful > 0:
        logger.info("Running feature importance analysis...")
        try:
            # Find best model checkpoint
            best_model = None
            for root, dirs, files in os.walk(experiment_dir):
                if "best_model.pt" in files:
                    best_model = os.path.join(root, "best_model.pt")
                    break
            
            if best_model:
                analysis_cmd = [
                    "python", "analyze_feature_importance.py",
                    "--checkpoint", best_model,
                    "--experiments-dir", str(experiment_dir),
                    "--output-dir", str(experiment_dir / "analysis")
                ]
                logger.info(f"Running analysis: {' '.join(analysis_cmd)}")
                subprocess.run(analysis_cmd)
                logger.info("Analysis completed")
            else:
                logger.warning("No best model found, skipping analysis")
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
    
    logger.info("All tasks completed")

if __name__ == "__main__":
    main()