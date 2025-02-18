"""
Configuration file for wildfire spread prediction model.
"""

config = {
    # Data paths
    'train_path': 'C:/data/train',
    'val_path': 'C:/data/val',
    'test_path': 'C:/data/test',
    'cache_dir': 'C:/data/cached_data',
    
    # Model parameters
    'n_channels': 11,  # NDVI, elevation, population, pdsi, vs, pr, sph, tmmx, th, tmmn, erc
    'n_classes': 1,    # FireMask
    
    # Training parameters
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'epochs': 100,
    'patience': 15,
    
    # Evaluation parameters
    'threshold': 0.5,
    
    # Paths for saving results
    'checkpoint_dir': 'C:/Users/footb/Capstone_Personal/checkpoints',
    'log_dir': 'C:/Users/footb/Capstone_Personal/logs',
    'evaluation_dir': 'C:/Users/footb/Capstone_Personal/evaluation_results'
}