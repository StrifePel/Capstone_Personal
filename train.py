import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
import json
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from data_utils import create_dataloaders
from model import UNet

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
        checkpoint_path: Optional[str] = None
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
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
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(self.config.get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.logger.info(f"Resumed training from epoch {self.start_epoch}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

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
    
    def train(self):
        """Main training loop."""
        try:
            self.logger.info(f"Starting training on device: {self.device}")
            self.logger.info(f"Training batches: {len(self.train_loader)}")
            self.logger.info(f"Validation batches: {len(self.val_loader)}")
            
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
                self.save_checkpoint(epoch, val_loss, is_best)
                
                # Early stopping
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.logger.info("Training completed")

def main():
    # Configuration
    config = {
        'train_path': './data/train',
        'val_path': './data/val',
        'cache_dir': './data/cached_data',
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 15,
        'min_delta': 1e-4,
        'max_grad_norm': 1.0,
        'n_channels': 11,
        'n_classes': 1,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs'
    }
    
    try:
        # Create cache directory
        os.makedirs(config['cache_dir'], exist_ok=True)
        
        # Check for CUDA
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available. Training will be slower.")
            print(f"PyTorch version: {torch.__version__}")
            
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            config['train_path'],
            config['val_path'],
            config['batch_size'],
            config['num_workers'],
            config['cache_dir']
        )
        
        # Initialize model
        model = UNet(
            n_channels=config['n_channels'],
            n_classes=config['n_classes']
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main()