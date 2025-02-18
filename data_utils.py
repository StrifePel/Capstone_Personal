import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import tensorflow as tf
import logging
from pathlib import Path
import numpy as np

class WildfireDataset(Dataset):
    """PyTorch Dataset for wildfire spread prediction with PT file caching."""
    def __init__(self, tfrecord_path: str, cache_dir: str = "cached_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
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
            pt_file = self.cache_dir / f"{Path(tfrecord_file).stem}.pt"
            
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
        """Convert TF features to PyTorch tensors."""
        try:
            # Process input features
            input_keys = [
                'NDVI', 'elevation', 'population', 'pdsi', 
                'vs', 'pr', 'sph', 'tmmx', 'th', 'tmmn', 'erc'
            ]
            
            input_tensors = []
            for key in input_keys:
                feature_data = features[key]  # Already in the correct format from _parse_tfrecord
                # Reshape to 64x64 if needed
                feature_data = tf.reshape(feature_data, (64, 64))
                input_tensors.append(torch.from_numpy(feature_data.numpy()).float())
            
            # Stack input features
            input_tensor = torch.stack(input_tensors)
            
            # Process target (FireMask)
            target_data = tf.reshape(features['FireMask'], (64, 64))
            target_tensor = torch.from_numpy(target_data.numpy()).float()
            
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
    cache_dir: str = "cached_data"
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with caching."""
    
    # Create cache directories
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(val_cache, exist_ok=True)
    
    # Create datasets
    logging.info("Creating TensorFlow dataset...")
    train_dataset = WildfireDataset(train_path, cache_dir=train_cache)
    val_dataset = WildfireDataset(val_path, cache_dir=val_cache)
    
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