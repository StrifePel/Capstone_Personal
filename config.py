"""
Configuration file for wildfire spread prediction model.
"""

config = {
    # Data paths
    'train_path': './data/train',
    'val_path': './data/val',
    'test_path': './data/test',
    'cache_dir': './data/cached_data',
    
    # Features configuration
    'features': {
        'NDVI': True,           # Normalized Difference Vegetation Index
        'elevation': True,      # Elevation data
        'population': True,     # Population density
        'pdsi': True,           # Palmer Drought Severity Index
        'vs': True,             # Wind speed
        'pr': True,             # Precipitation
        'sph': True,            # Specific humidity
        'tmmx': True,           # Maximum temperature
        'th': True,             # Wind direction
        'tmmn': True,           # Minimum temperature
        'erc': True,            # Energy Release Component (fire danger)
    },
    
    # Model parameters
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
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'evaluation_dir': './evaluation_results'
}

# Calculate number of active channels based on feature selection
def get_active_features():
    """Get list of active feature names."""
    return [feat for feat, is_active in config['features'].items() if is_active]

def get_n_channels():
    """Get number of active channels."""
    return sum(1 for feat, is_active in config['features'].items() if is_active)

# Set the number of channels dynamically
config['n_channels'] = get_n_channels()
config['active_features'] = get_active_features()