import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import logging
import json
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import argparse
import pickle
import traceback
import sys
import numpy as np
import tempfile

from data_utils import create_dataloaders
from model import UNet, FeatureImportanceAnalyzer
from config import config, get_active_features, get_n_channels

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        checkpoint_path: Optional[str] = None,
        run_id: Optional[str] = None,
        disable_checkpoints: bool = False
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.disable_checkpoints = disable_checkpoints
        
        # Store feature information
        self.active_features = config.get('active_features', [])
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Set cuda device options for optimal performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU. Training will be slower.")
            
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config.get('min_delta', 1e-4)
        )
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Setup logging and checkpointing
        self.setup_logging()
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints')) / self.run_id
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Test file system before starting
        self.check_file_system(str(self.checkpoint_dir))
        self.check_file_system("C:/temp")
        
        # Save config
        self.save_config()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{self.run_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log feature configuration
        self.logger.info(f"Training with features: {', '.join(self.active_features)}")
    
    def save_config(self):
        """Save the configuration used for this training run."""
        config_path = self.checkpoint_dir / "config.json"
        try:
            with open(config_path, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_config = {}
                for k, v in self.config.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) and k != 'device':
                        serializable_config[k] = v
                    else:
                        serializable_config[k] = str(v)
                json.dump(serializable_config, f, indent=2)
            self.logger.info(f"Config saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.logger.info(f"Resumed training from epoch {self.start_epoch}")
            
            # Check if feature configuration matches
            if 'active_features' in checkpoint and checkpoint['active_features'] != self.active_features:
                self.logger.warning(f"Feature mismatch! Checkpoint used: {checkpoint['active_features']}")
                self.logger.warning(f"Current features: {self.active_features}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint with debugging."""
        if self.disable_checkpoints:
            self.logger.info("Checkpoints disabled, skipping save")
            return

        try:
            self.logger.info(f"Preparing checkpoint data for epoch {epoch}")
            
            # Log memory usage before checkpoint creation
            if torch.cuda.is_available():
                self.logger.info(f"GPU memory: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, "
                                f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
                
            # Create checkpoint dict
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss,
                'active_features': self.active_features,
                'config': self.config
            }
            
            # Log checkpoint size estimation
            try:
                model_size_mb = sys.getsizeof(pickle.dumps(self.model.state_dict())) / (1024 * 1024)
                optimizer_size_mb = sys.getsizeof(pickle.dumps(self.optimizer.state_dict())) / (1024 * 1024)
                self.logger.info(f"Estimated sizes - Model: {model_size_mb:.2f}MB, Optimizer: {optimizer_size_mb:.2f}MB")
            except Exception as e:
                self.logger.warning(f"Could not estimate checkpoint size: {e}")
            
            # Log path info
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            self.logger.info(f"Saving checkpoint to: {checkpoint_path} (path length: {len(str(checkpoint_path))} chars)")
            
            # Try various saving methods
            save_methods = [
                (self._save_method_standard, checkpoint_path, "Standard save"),
                (self._save_method_legacy, checkpoint_path, "Legacy pickle save"),
                (self._save_method_state_dict_only, checkpoint_path, "State dict only save"),
                (self._save_method_temp_path, checkpoint, "Temp path save"),
                (self._save_method_numpy, checkpoint_path, "Numpy arrays save")
            ]
            
            success = False
            for save_func, path, method_name in save_methods:
                try:
                    self.logger.info(f"Trying {method_name}...")
                    result = save_func(path if path is not checkpoint else checkpoint)
                    if result:
                        self.logger.info(f"Successfully saved using {method_name}")
                        success = True
                        break
                except Exception as e:
                    self.logger.warning(f"{method_name} failed: {e}")
            
            if not success:
                self.logger.error("All save methods failed")
                
            # Save best model separately if this is the best model
            if is_best and success:
                try:
                    best_path = self.checkpoint_dir / "best_model.pt"
                    self.logger.info(f"Saving best model to: {best_path}")
                    if self._save_method_standard(best_path, checkpoint):
                        self.logger.info(f"Best model saved with validation loss: {val_loss:.4f}")
                    else:
                        self.logger.warning("Could not save best model with standard method, trying alternatives")
                        for save_func, _, method_name in save_methods[1:]:  # Skip standard method
                            try:
                                result = save_func(best_path if type(best_path) is not dict else checkpoint)
                                if result:
                                    self.logger.info(f"Best model saved using {method_name}")
                                    break
                            except Exception as e:
                                self.logger.warning(f"{method_name} failed for best model: {e}")
                except Exception as e:
                    self.logger.error(f"Error saving best model: {e}")
                
        except Exception as e:
            self.logger.error(f"Error in save_checkpoint: {e}")
            self.logger.error(traceback.format_exc())
    
    def _save_method_standard(self, path, checkpoint=None):
        """Standard torch.save method."""
        if checkpoint is None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
            }
        torch.save(checkpoint, path)
        return True
    
    def _save_method_legacy(self, path, checkpoint=None):
        """Legacy pickle save method."""
        if checkpoint is None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
            }
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
        return True
    
    def _save_method_state_dict_only(self, path):
        """Save only the model state dict."""
        torch.save(self.model.state_dict(), str(path).replace('.pt', '_state_dict.pt'))
        return True
    
    def _save_method_temp_path(self, checkpoint):
        """Save to a temporary path."""
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "temp_checkpoint.pt")
        self.logger.info(f"Saving to temporary path: {tmp_path}")
        torch.save(checkpoint, tmp_path)
        return True
    
    def _save_method_numpy(self, path):
        """Save as numpy arrays."""
        state_dict = self.model.state_dict()
        np_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        np.savez(str(path).replace('.pt', '.npz'), **np_dict)
        return True

    def check_file_system(self, path):
        """Check if the file system is working properly."""
        self.logger.info(f"Checking file system at: {path}")
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Try to create a small test file
            test_file = os.path.join(path, "test_file.txt")
            with open(test_file, 'w') as f:
                f.write("Test content")
            self.logger.info(f"Successfully wrote small test file: {test_file}")
            
            # Try to create a larger test file (10MB)
            large_test_file = os.path.join(path, "large_test_file.bin")
            with open(large_test_file, 'wb') as f:
                f.write(b'0' * 10 * 1024 * 1024)  # 10MB of zeros
            self.logger.info(f"Successfully wrote 10MB test file: {large_test_file}")
            
            # Clean up
            os.remove(test_file)
            os.remove(large_test_file)
            self.logger.info("Test files cleaned up")
            
            return True
        except Exception as e:
            self.logger.error(f"File system test failed: {e}")
            return False
    
    def debug_save(self):
        """Debug PyTorch saving with different approaches."""
        self.logger.info("Starting save debugging...")
        
        model_state = self.model.state_dict()
        
        # Test locations
        locations = [
            self.checkpoint_dir / "debug_model.pt",
            Path("C:/temp/debug_model.pt"),
            Path(f"{os.path.expanduser('~')}/debug_model.pt")  # Home directory
        ]
        
        # Test saving methods
        for loc in locations:
            self.logger.info(f"Trying to save to: {loc}")
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(loc), exist_ok=True)
                
                # Method 1: Standard save
                torch.save(model_state, loc)
                self.logger.info(f"SUCCESS: Standard save worked at {loc}")
                break
            except Exception as e:
                self.logger.error(f"Standard save failed: {e}")
                
                try:
                    # Method 2: Legacy pickling
                    torch.save(model_state, str(loc), _use_new_zipfile_serialization=False)
                    self.logger.info(f"SUCCESS: Legacy pickle save worked at {loc}")
                    break
                except Exception as e2:
                    self.logger.error(f"Legacy pickle save failed: {e2}")
                    
                    try:
                        # Method 3: Save as numpy arrays
                        import numpy as np
                        np_dict = {k: v.cpu().numpy() for k, v in model_state.items()}
                        np.savez(str(loc).replace('.pt', '.npz'), **np_dict)
                        self.logger.info(f"SUCCESS: Numpy save worked at {loc}")
                        break
                    except Exception as e3:
                        self.logger.error(f"Numpy save failed: {e3}")
        
        self.logger.info("Save debugging completed")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_count = len(self.train_loader)
        
        start_time = time.time()
        last_update = start_time
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    # Move data to device
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    
                    self.optimizer.step()
                    
                    # Update statistics
                    total_loss += loss.item()
                    
                    # Update progress bar
                    current_time = time.time()
                    if current_time - last_update > 10:  # Update every 10 seconds
                        elapsed = current_time - start_time
                        speed = (batch_idx + 1) / elapsed
                        remaining = (batch_count - batch_idx - 1) / speed
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                            'speed': f"{speed:.1f} it/s",
                            'eta': f"{remaining/60:.1f}min"
                        })
                        last_update = current_time
                        
                except Exception as e:
                    self.logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
                
        return total_loss / batch_count

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch_idx, (data, target) in enumerate(pbar):
                    try:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        total_loss += loss.item()
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                        continue
        
        return total_loss / len(self.val_loader)
    
    def analyze_feature_importance(self):
        """Analyze feature importance after training with robust error handling."""
        if not self.active_features:
            self.logger.warning("No feature names provided, skipping importance analysis")
            return
            
        try:
            self.logger.info("Analyzing feature importance...")
            analyzer = FeatureImportanceAnalyzer(self.model, self.active_features)
            importance = analyzer.analyze_gradient_based_importance(self.val_loader, self.device)
            
            self.logger.info("Feature importance ranking:")
            for feature, score in importance:
                self.logger.info(f"  {feature}: {score:.6f}")
                
            # Try to save to file, but don't crash if it fails
            try:
                with open(self.checkpoint_dir / "feature_importance.json", 'w') as f:
                    json.dump({feat: float(score) for feat, score in importance}, f, indent=2)
                self.logger.info("Saved feature importance to file")
            except Exception as e:
                self.logger.warning(f"Could not save importance file: {e}")
                try:
                    # Try alternate location
                    os.makedirs("C:/temp", exist_ok=True)
                    with open("C:/temp/feature_importance.json", 'w') as f:
                        json.dump({feat: float(score) for feat, score in importance}, f, indent=2)
                    self.logger.info("Saved feature importance to C:/temp/")
                except Exception as e2:
                    self.logger.warning(f"Could not save to alternate location: {e2}")
                    
            return importance
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def train(self):
        """Main training loop."""
        try:
            self.logger.info(f"Starting training on device: {self.device}")
            self.logger.info(f"Training batches: {len(self.train_loader)}")
            self.logger.info(f"Validation batches: {len(self.val_loader)}")
            
            # Run debug save before starting training
            self.debug_save()
            
            for epoch in range(self.start_epoch, self.config['epochs']):
                self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                
                # Training phase
                train_loss = self.train_epoch()
                self.logger.info(f"Training Loss: {train_loss:.4f}")
                
                # Validation phase
                val_loss = self.validate()
                self.logger.info(f"Validation Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if not self.disable_checkpoints:
                    self.save_checkpoint(epoch, val_loss, is_best)
                
                # Early stopping
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Training completed")
            # Analyze feature importance at the end of training
            if self.model.training:
                self.model.eval()
            self.analyze_feature_importance()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train wildfire prediction model')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                      help='Specific features to use (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint for resuming training')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save output (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                      help='Learning rate (overrides config)')
    parser.add_argument('--early-stopping', type=int, default=None,
                      help='Early stopping patience (overrides config)')
    parser.add_argument('--disable-checkpoints', action='store_true',
                      help='Disable saving checkpoints (use for troubleshooting)')
    
    return parser.parse_args()


def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Load base configuration
    train_config = config.copy()
    
    # Override with command line arguments
    if args.features:
        train_config['active_features'] = args.features
        train_config['n_channels'] = len(args.features)
    
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    
    if args.epochs:
        train_config['epochs'] = args.epochs
    
    if args.learning_rate:
        train_config['learning_rate'] = args.learning_rate
    
    if args.early_stopping:
        train_config['patience'] = args.early_stopping
    
    if args.output_dir:
        train_config['checkpoint_dir'] = args.output_dir
    
    # Create dataloaders
    active_features = train_config.get('active_features', get_active_features())
    train_loader, val_loader = create_dataloaders(
        train_config['train_path'],
        train_config['val_path'],
        train_config['batch_size'],
        train_config['num_workers'],
        train_config['cache_dir'],
        active_features=active_features
    )
    
    # Initialize model
    model = UNet(
        n_channels=len(active_features),
        n_classes=train_config['n_classes'],
        feature_names=active_features
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        checkpoint_path=args.checkpoint,
        disable_checkpoints=args.disable_checkpoints
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()