import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import tensorflow as tf
import logging
from pathlib import Path
import numpy as np

class WildfireDataset(Dataset):
    """PyTorch Dataset for wildfire spread prediction with PT file caching."""
    def __init__(self, tfrecord_path: str, cache_dir: str = "cached_data", active_features: List[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # All available features
        self.all_features = [
            'NDVI', 'elevation', 'population', 'pdsi', 
            'vs', 'pr', 'sph', 'tmmx', 'th', 'tmmn', 'erc'
        ]
        
        # Active features to use
        self.active_features = active_features if active_features else self.all_features
        self.feature_indices = {feat: i for i, feat in enumerate(self.all_features)}
        
        # Generate a unique cache identifier based on active features
        self.feature_hash = "_".join(sorted(self.active_features))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Active features: {self.active_features}")
        
        # Get all TFRecord filenames
        if os.path.isdir(tfrecord_path):
            self.tfrecord_files = [os.path.join(tfrecord_path, f) for f in os.listdir(tfrecord_path) 
                                 if f.endswith('.tfrecord')]
            self.logger.info(f"Looking for TFRecord files in: {tfrecord_path}")
        else:
            self.tfrecord_files = [tfrecord_path]
            
        self.examples = []
        self.length = 0
        
        # Convert and cache each TFRecord file
        for tfrecord_file in self.tfrecord_files:
            self.logger.info(f"Found TFRecord file: {tfrecord_file}")
            # Include feature selection in cache filename
            pt_file = self.cache_dir / f"{Path(tfrecord_file).stem}_{self.feature_hash}.pt"
            
            if pt_file.exists():
                self.logger.info(f"Loading cached PT file: {pt_file}")
                cached_data = torch.load(pt_file)
                self.examples.extend(cached_data)
            else:
                self.logger.info(f"Converting TFRecord to PT: {tfrecord_file}")
                # Create dataset for this file
                dataset = tf.data.TFRecordDataset([tfrecord_file])
                dataset = dataset.map(self._parse_tfrecord)
                
                # Convert examples to PyTorch tensors and save
                file_examples = []
                for features in dataset.as_numpy_iterator():
                    try:
                        processed_features = self._process_features(features)
                        file_examples.append(processed_features)
                    except Exception as e:
                        self.logger.warning(f"Error processing feature: {str(e)}")
                        continue
                
                # Save to cache if we processed any features successfully
                if file_examples:
                    torch.save(file_examples, pt_file)
                    self.examples.extend(file_examples)
                else:
                    self.logger.error(f"No valid examples processed from {tfrecord_file}")
            
        self.length = len(self.examples)
        self.logger.info(f"Successfully loaded {self.length} examples")
    
    def _parse_tfrecord(self, example_proto):
        """Parse TFRecord example."""
        feature_description = {
            'NDVI': tf.io.FixedLenFeature([4096], tf.float32),
            'PrevFireMask': tf.io.FixedLenFeature([4096], tf.float32),
            'elevation': tf.io.FixedLenFeature([4096], tf.float32),
            'population': tf.io.FixedLenFeature([4096], tf.float32),
            'FireMask': tf.io.FixedLenFeature([4096], tf.float32),
            'pdsi': tf.io.FixedLenFeature([4096], tf.float32),
            'vs': tf.io.FixedLenFeature([4096], tf.float32),
            'pr': tf.io.FixedLenFeature([4096], tf.float32),
            'sph': tf.io.FixedLenFeature([4096], tf.float32),
            'tmmx': tf.io.FixedLenFeature([4096], tf.float32),
            'th': tf.io.FixedLenFeature([4096], tf.float32),
            'tmmn': tf.io.FixedLenFeature([4096], tf.float32),
            'erc': tf.io.FixedLenFeature([4096], tf.float32)
        }
        
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def _process_features(self, features):
        """Convert TF features to PyTorch tensors using only active features."""
        try:
            # Process only active input features
            input_tensors = []
            for key in self.active_features:
                if key in features:
                    feature_data = features[key]
                    # Reshape to 64x64 
                    feature_data = tf.reshape(feature_data, (64, 64))
                    # Convert to PyTorch tensor
                    feature_tensor = torch.from_numpy(feature_data.numpy()).float()
                    # Handle invalid values
                    feature_tensor = torch.nan_to_num(feature_tensor, 0.0)
                    input_tensors.append(feature_tensor)
                else:
                    self.logger.warning(f"Feature {key} not found in TFRecord")
            
            # Stack input features
            input_tensor = torch.stack(input_tensors)
            
            # Process target (FireMask)
            target_data = tf.reshape(features['FireMask'], (64, 64))
            target_tensor = torch.from_numpy(target_data.numpy()).float()
            
            # Handle invalid values and clamp to [0, 1]
            target_tensor = torch.nan_to_num(target_tensor, 0.0)
            target_tensor = torch.clamp(target_tensor, 0.0, 1.0)
            
            # Add channel dimension
            target_tensor = target_tensor.unsqueeze(0)
            
            return (input_tensor, target_tensor)
            
        except Exception as e:
            self.logger.error(f"Error processing features: {str(e)}")
            raise
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]

def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: str = "cached_data",
    active_features: List[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with caching."""
    
    # Create cache directories
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(val_cache, exist_ok=True)
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = WildfireDataset(train_path, cache_dir=train_cache, active_features=active_features)
    val_dataset = WildfireDataset(val_path, cache_dir=val_cache, active_features=active_features)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader