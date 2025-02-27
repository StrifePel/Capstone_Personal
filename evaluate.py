import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import logging
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

from data_utils import WildfireDataset
from model import UNet
from config import config, get_active_features

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        thresholds: List[float] = None,
        active_features: List[str] = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.thresholds = thresholds if thresholds is not None else [0.5]
        self.active_features = active_features
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.active_features:
            self.logger.info(f"Evaluating with features: {', '.join(self.active_features)}")
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Flatten arrays for metric calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        y_prob_flat = y_prob.reshape(-1)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat,
            y_pred_flat,
            average='binary'
        )
        
        # Calculate ROC AUC
        auc_roc = roc_auc_score(y_true_flat, y_prob_flat)
        
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
        predictions: Dict[float, torch.Tensor],
        save_path: Path
    ):
        """Plot sample predictions for different thresholds."""
        num_thresholds = len(predictions)
        fig, axes = plt.subplots(3, num_thresholds + 2, figsize=(5*(num_thresholds + 2), 15))
        
        for i in range(3):
            # Plot input features
            axes[i, 0].imshow(inputs[i].mean(dim=0).cpu())
            axes[i, 0].set_title(f'Input Features\n(Sample {i+1})')
            plt.colorbar(axes[i, 0].images[0], ax=axes[i, 0])
            
            # Plot ground truth
            axes[i, 1].imshow(targets[i].squeeze(0).cpu(), cmap='hot')
            axes[i, 1].set_title(f'Ground Truth\n(Sample {i+1})')
            plt.colorbar(axes[i, 1].images[0], ax=axes[i, 1])
            
            # Plot predictions for each threshold
            for j, (threshold, preds) in enumerate(predictions.items()):
                axes[i, j+2].imshow(preds[i].squeeze(0).cpu(), cmap='hot')
                axes[i, j+2].set_title(f'Prediction\nThreshold={threshold:.2f}')
                plt.colorbar(axes[i, j+2].images[0], ax=axes[i, j+2])
        
        plt.tight_layout()
        plt.savefig(save_path / 'prediction_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_metrics(
        self,
        metrics: Dict[float, Dict[str, float]],
        save_path: Path
    ):
        """Plot metrics vs threshold."""
        thresholds = list(metrics.keys())
        precision = [m['precision'] for m in metrics.values()]
        recall = [m['recall'] for m in metrics.values()]
        f1 = [m['f1_score'] for m in metrics.values()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision, 'b-', label='Precision')
        plt.plot(thresholds, recall, 'r-', label='Recall')
        plt.plot(thresholds, f1, 'g-', label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path / 'threshold_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, save_dir: Path) -> Dict[float, Dict[str, float]]:
        """Evaluate the model on test set with multiple thresholds."""
        self.model.eval()
        save_dir.mkdir(exist_ok=True)
        
        # First pass: collect all outputs and targets
        all_targets = []
        all_outputs = []
        first_batch_inputs = None
        first_batch_targets = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Collecting predictions")):
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Store outputs and targets
                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())
                
                # Store first batch for visualization
                if batch_idx == 0:
                    first_batch_inputs = data
                    first_batch_targets = target
        
        # Concatenate all outputs and targets
        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        # Calculate metrics for each threshold
        metrics = {}
        predictions = {}
        
        for threshold in self.thresholds:
            # Apply threshold
            predictions[threshold] = (all_outputs > threshold).astype(float)
            
            # Calculate metrics
            metrics[threshold] = self.calculate_metrics(
                all_targets,
                predictions[threshold],
                all_outputs
            )
            
            # Log results
            self.logger.info(f"\nResults for threshold {threshold:.2f}:")
            for metric_name, value in metrics[threshold].items():
                self.logger.info(f"{metric_name}: {value:.4f}")
        
        # Plot predictions for first batch with different thresholds
        first_batch_predictions = {
            threshold: torch.tensor(predictions[threshold][:first_batch_inputs.shape[0]])
            for threshold in self.thresholds
        }
        self.plot_predictions(
            first_batch_inputs,
            first_batch_targets,
            first_batch_predictions,
            save_dir
        )
        
        # Plot metrics vs threshold
        self.plot_threshold_metrics(metrics, save_dir)
        
        # Save feature configuration
        if self.active_features:
            with open(save_dir / 'features.json', 'w') as f:
                json.dump({'active_features': self.active_features}, f, indent=2)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate wildfire prediction model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='./data/test',
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Thresholds for evaluation')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                        help='Specific features to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Define thresholds to test
    thresholds = args.thresholds
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
        
        # Get active features from checkpoint or argument
        if args.features:
            active_features = args.features
            print(f"Using specified features: {active_features}")
        elif 'active_features' in checkpoint:
            active_features = checkpoint['active_features']
            print(f"Using features from checkpoint: {active_features}")
        else:
            active_features = get_active_features()
            print(f"Using features from config: {active_features}")
            
        # Determine number of channels
        n_channels = len(active_features)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataloader
    try:
        test_dataset = WildfireDataset(
            tfrecord_path=args.test_path,
            cache_dir=config['cache_dir'],
            active_features=active_features
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"Created test dataloader with {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error creating test dataloader: {e}")
        return
    
    # Initialize model
    model = UNet(
        n_channels=n_channels,
        n_classes=1,
        feature_names=active_features
    )
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("Loaded model weights successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Initialize evaluator with multiple thresholds
    evaluator = Evaluator(
        model, 
        test_loader, 
        device, 
        thresholds=thresholds,
        active_features=active_features
    )
    print(f"Initialized evaluator with thresholds: {thresholds}")
    
    # Run evaluation
    try:
        save_dir = Path(args.output_dir)
        if active_features:
            # Create a specific directory for this feature configuration
            feature_hash = "_".join(sorted(active_features))
            save_dir = save_dir / feature_hash
        save_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving results to {save_dir}")
        
        metrics = evaluator.evaluate(save_dir)
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {
                str(threshold): {
                    k: float(v) for k, v in threshold_metrics.items()
                }
                for threshold, threshold_metrics in metrics.items()
            }
            json.dump(metrics_json, f, indent=4)
        print(f"Saved metrics to {save_dir / 'metrics.json'}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

if __name__ == "__main__":
    main()