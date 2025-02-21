import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_utils import create_dataloaders
from model import WildfirePredictor
import logging

class ModelDiagnostics:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = WildfirePredictor(n_channels=config['n_channels']).to(self.device)
        checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pt'
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Model loaded successfully")
        else:
            self.logger.error(f"No checkpoint found at {checkpoint_path}")
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Create dataloader
        _, self.loader = create_dataloaders(
            config['test_path'],
            config['test_path'],
            batch_size=1,  # Use batch size 1 for detailed analysis
            num_workers=0  # No multiprocessing for debugging
        )

    def analyze_data_distribution(self):
        """Analyze the distribution of values in the input data and targets."""
        self.logger.info("Analyzing data distribution...")
        
        sample_features = []
        sample_targets = []
        
        for i, (data, weather, target) in enumerate(self.loader):
            if i >= 10:  # Analyze first 10 batches
                break
                
            # Log data statistics
            self.logger.info(f"\nBatch {i}:")
            self.logger.info(f"Input shape: {data.shape}")
            self.logger.info(f"Weather shape: {weather.shape}")
            self.logger.info(f"Target shape: {target.shape}")
            
            self.logger.info(f"Input range: [{data.min():.4f}, {data.max():.4f}]")
            self.logger.info(f"Weather range: [{weather.min():.4f}, {weather.max():.4f}]")
            self.logger.info(f"Target unique values: {torch.unique(target).tolist()}")
            
            # Store for distribution analysis
            sample_features.append(data.numpy())
            sample_targets.append(target.numpy())
            
            # Save visualization
            self._plot_sample(data[0], target[0], i)
        
        return sample_features, sample_targets

    def analyze_model_outputs(self):
        """Analyze the model's output distribution and activation values."""
        self.logger.info("\nAnalyzing model outputs...")
        self.model.eval()
        
        outputs_stats = []
        with torch.no_grad():
            for i, (data, weather, target) in enumerate(self.loader):
                if i >= 10:  # Analyze first 10 batches
                    break
                
                # Move to device
                data = data.to(self.device)
                weather = weather.to(self.device)
                target = target.to(self.device)
                
                # Get model output
                output = self.model(data, weather)
                probs = torch.sigmoid(output)
                
                # Log statistics
                self.logger.info(f"\nBatch {i}:")
                self.logger.info(f"Raw output range: [{output.min():.4f}, {output.max():.4f}]")
                self.logger.info(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
                self.logger.info(f"Mean probability: {probs.mean():.4f}")
                
                # Calculate prediction stats
                preds = (probs > 0.5).float()
                true_positives = ((preds == 1) & (target == 1)).sum().item()
                false_positives = ((preds == 1) & (target == 0)).sum().item()
                self.logger.info(f"True positives: {true_positives}")
                self.logger.info(f"False positives: {false_positives}")
                
                # Store statistics
                outputs_stats.append({
                    'output_min': output.min().item(),
                    'output_max': output.max().item(),
                    'prob_min': probs.min().item(),
                    'prob_max': probs.max().item(),
                    'prob_mean': probs.mean().item(),
                    'true_positives': true_positives,
                    'false_positives': false_positives
                })
                
                # Save visualization
                self._plot_predictions(data[0], target[0], preds[0], probs[0], i)
        
        return outputs_stats

    def analyze_model_parameters(self):
        """Analyze the model's parameters and gradients."""
        self.logger.info("\nAnalyzing model parameters...")
        
        total_params = 0
        for name, param in self.model.named_parameters():
            param_size = param.numel()
            total_params += param_size
            
            self.logger.info(f"\nParameter: {name}")
            self.logger.info(f"Shape: {param.shape}")
            self.logger.info(f"Size: {param_size}")
            self.logger.info(f"Mean: {param.mean():.4f}")
            self.logger.info(f"Std: {param.std():.4f}")
            self.logger.info(f"Min: {param.min():.4f}")
            self.logger.info(f"Max: {param.max():.4f}")
            
            # Check for potential issues
            if torch.isnan(param).any():
                self.logger.error(f"NaN values found in {name}")
            if torch.isinf(param).any():
                self.logger.error(f"Inf values found in {name}")
        
        self.logger.info(f"\nTotal parameters: {total_params:,}")

    def _plot_sample(self, features, target, index):
        """Plot input features and target for visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot mean of input features
        ax1.imshow(features.mean(dim=0))
        ax1.set_title('Input Features (Mean)')
        
        # Plot target
        ax2.imshow(target.squeeze())
        ax2.set_title('Target')
        
        plt.savefig(f'diagnostic_sample_{index}.png')
        plt.close()

    def _plot_predictions(self, features, target, prediction, probabilities, index):
        """Plot model predictions with input and target."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot input features
        ax1.imshow(features.mean(dim=0).cpu())
        ax1.set_title('Input Features (Mean)')
        
        # Plot target
        ax2.imshow(target.cpu().squeeze())
        ax2.set_title('Target')
        
        # Plot prediction
        ax3.imshow(prediction.cpu().squeeze())
        ax3.set_title('Prediction')
        
        # Plot probabilities
        im = ax4.imshow(probabilities.cpu().squeeze())
        ax4.set_title('Probability Map')
        plt.colorbar(im, ax=ax4)
        
        plt.savefig(f'diagnostic_prediction_{index}.png')
        plt.close()

def main():
    config = {
        'test_path': './data/test',
        'batch_size': 1,
        'num_workers': 0,
        'n_channels': 11,
        'n_classes': 1,
        'checkpoint_dir': './checkpoints',
    }
    
    diagnostics = ModelDiagnostics(config)
    
    # Run diagnostics
    print("\n=== Starting Diagnostics ===")
    
    print("\n1. Analyzing Data Distribution...")
    features, targets = diagnostics.analyze_data_distribution()
    
    print("\n2. Analyzing Model Outputs...")
    output_stats = diagnostics.analyze_model_outputs()
    
    print("\n3. Analyzing Model Parameters...")
    diagnostics.analyze_model_parameters()
    
    print("\n=== Diagnostics Complete ===")
    print("Check the log file and generated visualizations for detailed results.")

if __name__ == '__main__':
    main()