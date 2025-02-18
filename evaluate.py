import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import logging
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import create_dataloaders
from model import UNet

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        threshold: float = 0.5
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.flatten(),
            y_pred.flatten(),
            average='binary'
        )
        
        auc_roc = roc_auc_score(y_true.flatten(), y_prob.flatten())
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
    
    def plot_predictions(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        save_path: Path
    ):
        """Plot sample predictions."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        for i in range(3):
            # Plot input features (composite)
            axes[i, 0].imshow(inputs[i].mean(dim=0).cpu())
            axes[i, 0].set_title(f'Input Features (Sample {i+1})')
            
            # Plot ground truth
            axes[i, 1].imshow(targets[i].cpu(), cmap='hot')
            axes[i, 1].set_title(f'Ground Truth (Sample {i+1})')
            
            # Plot prediction
            axes[i, 2].imshow(predictions[i].cpu(), cmap='hot')
            axes[i, 2].set_title(f'Prediction (Sample {i+1})')
        
        plt.tight_layout()
        plt.savefig(save_path / 'prediction_samples.png')
        plt.close()
    
    def evaluate(self, save_dir: Path) -> Dict[str, float]:
        """Evaluate the model on test set."""
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        save_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Store predictions and targets
                predictions = (output > self.threshold).float()
                all_targets.append(target.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(output.cpu().numpy())
                
                # Plot first batch
                if batch_idx == 0:
                    self.plot_predictions(
                        data,
                        target,
                        predictions,
                        save_dir
                    )
        
        # Concatenate all predictions and targets
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            all_targets,
            all_predictions,
            all_probabilities
        )
        
        # Log results
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics

def main():
    # Load configuration
    checkpoint = torch.load('path/to/best/checkpoint.pt')
    config = checkpoint['config']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataloader with cache
    _, test_loader = create_dataloaders(
        config['train_path'],  # Not used
        config['test_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        cache_dir=config['cache_dir']
    )
    
    # Initialize model
    model = UNet(
        n_channels=config['n_channels'],
        n_classes=config['n_classes']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Initialize evaluator
    evaluator = Evaluator(model, test_loader, device)
    
    # Run evaluation
    save_dir = Path(config['evaluation_dir'])
    metrics = evaluator.evaluate(save_dir)
    
    # Save metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()