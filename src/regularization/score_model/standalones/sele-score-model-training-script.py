"""
Score-Based Diffusion Model Training Script

This module implements training for a score-based generative model using
variance preserving SDE with a linear beta schedule.
"""

import logging
import time
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 100
    target_length: int = 500 # If lower than data's length, data will be downsampled.
    data_path: str = 'Data/sele_simulated_1000_curves_500_long.mat' # From a modified MATLAB code create_training_set.m with seed rng(12)
    output_path: str = 'Data/sele_score_net_d500.pt'
    beta_min: float = 0.1
    beta_max: float = 20.0
    time_eps: float = 1e-4
    # TODO: Add hidden_dims here. For 500 set to (512, 1024, 2048, 2048, 1024, 512)


class ScoreNetwork(nn.Module):
    """
    Score network for diffusion model.
    
    Takes as input the noisy data concatenated with time and outputs
    the score (gradient of log probability).
    
    Args:
        input_dim: Dimension of input data (data_dim + 1 for time)
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output (same as data_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 33,
        hidden_dims: Tuple[int, ...] = (512, 1024, 2048, 2048, 1024, 512), # For 32 set to: (64, 128, 256, 256, 128, 64),
        output_dim: int = 32
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Softplus()  # More stable than LogSigmoid
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the score network.
        
        Args:
            x: Noisy data tensor of shape (batch_size, data_dim)
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            Score tensor of shape (batch_size, data_dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.network(xt)


class DiffusionModel:
    """
    Variance Preserving SDE diffusion model with linear beta schedule.
    
    Args:
        score_network: Neural network that predicts the score
        config: Training configuration
        device: Device to run training on
    """
    
    def __init__(
        self,
        score_network: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.score_network = score_network
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_network.to(self.device)
        
        logger.info(f"Initialized diffusion model on device: {self.device}")
    
    def compute_diffusion_params(
        self,
        t: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for the diffusion process at time t.
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            x: Original data tensor of shape (batch_size, data_dim)
            
        Returns:
            Tuple of (mu_t, var_t) representing mean and variance
        """
        # Integral of beta(t) = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
        int_beta = (self.config.beta_min + 0.5 * (self.config.beta_max - self.config.beta_min) * t) * t
        
        mu_t = x * torch.exp(-0.5 * int_beta)
        var_t = -torch.expm1(-int_beta)  # 1 - exp(-int_beta), more numerically stable
        
        return mu_t, var_t
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the denoising score matching loss.
        
        Args:
            x: Clean data tensor of shape (batch_size, data_dim)
            
        Returns:
            Scalar loss tensor
        """
        batch_size = x.shape[0]
        
        # Sample random time steps
        t = torch.rand(
            (batch_size, 1),
            dtype=x.dtype,
            device=x.device
        ) * (1 - self.config.time_eps) + self.config.time_eps
        
        # Compute forward diffusion
        mu_t, var_t = self.compute_diffusion_params(t, x)
        
        # Sample noisy data
        noise = torch.randn_like(x)
        x_t = mu_t + torch.sqrt(var_t) * noise
        
        # True score (gradient of log probability)
        grad_log_p = -(x_t - mu_t) / var_t
        
        # Predicted score
        score = self.score_network(x_t, t)
        
        # Weighted MSE loss
        loss = (score - grad_log_p) ** 2
        weighted_loss = var_t * loss
        
        return weighted_loss.mean()
    
    def train_epoch(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            optimizer: Optimizer for updating weights
            
        Returns:
            Average loss for the epoch
        """
        self.score_network.train()
        total_loss = 0.0
        num_samples = 0
        
        for batch, in data_loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.score_network.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * batch.shape[0]
            num_samples += batch.shape[0]
        
        return total_loss / num_samples


def load_and_preprocess_data(
    data_path: str,
    target_length: int
) -> Tuple[torch.Tensor, float, float]:
    """
    Load and preprocess data from .mat file.
    
    Args:
        data_path: Path to .mat file
        target_length: Target length for downsampling
        
    Returns:
        Tuple of (preprocessed_data, original_min, original_max)
    """
    logger.info(f"Loading data from {data_path}")
    
    mat = scipy.io.loadmat(data_path)
    data = torch.tensor(mat['data'], dtype=torch.float32)
    
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Store original min/max for potential inverse transform
    data_min = data.min().item() # Originally 2.7389012e-21
    data_max = data.max().item() # Originally 0.03475773
    logger.info(f"Loaded data with min: {data_min}, max: {data_max}. This is important for normalization!")
    
    # Normalize to [-1, 1]
    data = 2 * (data - data_min) / (data_max - data_min) - 1
    
    # Downsample to target length
    original_len = data.shape[1]
    if original_len != target_length:
        indices = torch.linspace(0, original_len - 1, target_length).round().long()
        data = data[:, indices]
        logger.info(f"Downsampled data from {original_len} to {target_length} points")
    
    return data, data_min, data_max


def create_data_loader(
    data: torch.Tensor,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader from tensor data.
    
    Args:
        data: Data tensor
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(config: TrainingConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    data, data_min, data_max = load_and_preprocess_data(
        config.data_path,
        config.target_length
    )
    
    # Create data loader
    data_loader = create_data_loader(data, config.batch_size)
    
    # Initialize model
    score_network = ScoreNetwork(
        input_dim=config.target_length + 1,
        output_dim=config.target_length
    )
    
    diffusion_model = DiffusionModel(score_network, config)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        score_network.parameters(),
        lr=config.learning_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        epoch_loss = diffusion_model.train_epoch(data_loader, optimizer)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Time: {elapsed_time:.2f}s"
        )
    
    # Save model
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': score_network.state_dict(),
        'config': config,
        'data_min': data_min,
        'data_max': data_max,
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Total training time: {time.time() - start_time:.2f}s")


def main():
    """Main entry point."""
    config = TrainingConfig()
    train_model(config)


if __name__ == "__main__":
    main()