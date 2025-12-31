"""
GAIN: Generative Adversarial Imputation Networks
Adapted for multivariate time series data imputation in water quality domain.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Generator(nn.Module):
    """Generator network for GAIN."""
    
    def __init__(self, input_dim, hidden_dim=256):
        """
        Initialize Generator.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
        """
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        """
        Forward pass.
        
        Args:
            x: Input data with missing values
            mask: Binary mask (1 for observed, 0 for missing)
            
        Returns:
            Imputed data
        """
        # Concatenate data and mask as input
        inputs = torch.cat([x, mask], dim=1)
        # Generate imputed values
        imputed = self.network(inputs)
        # Combine observed and imputed values
        output = mask * x + (1 - mask) * imputed
        return output


class Discriminator(nn.Module):
    """Discriminator network for GAIN."""
    
    def __init__(self, input_dim, hidden_dim=256):
        """
        Initialize Discriminator.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
        """
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, hint):
        """
        Forward pass.
        
        Args:
            x: Imputed data
            hint: Hint vector for training
            
        Returns:
            Predicted mask
        """
        # Concatenate data and hint as input
        inputs = torch.cat([x, hint], dim=1)
        # Predict which values are observed
        output = self.network(inputs)
        return output


class GAIN:
    """Generative Adversarial Imputation Networks."""
    
    def __init__(self, input_dim, hidden_dim=256, alpha=100, hint_rate=0.9, 
                 learning_rate=0.001, device='cpu'):
        """
        Initialize GAIN.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            alpha: Hyperparameter for loss balance
            hint_rate: Hint rate for training
            learning_rate: Learning rate for optimizers
            device: Device to use ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.device = device
        
        # Initialize networks
        self.generator = Generator(input_dim, hidden_dim).to(device)
        self.discriminator = Discriminator(input_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def _generate_hint(self, mask, hint_rate):
        """
        Generate hint matrix.
        
        Args:
            mask: Binary mask
            hint_rate: Probability of providing hints
            
        Returns:
            Hint matrix
        """
        hint = torch.rand_like(mask).to(self.device)
        hint = (hint < hint_rate).float()
        hint = hint * mask + 0.5 * (1 - hint)
        return hint
    
    def train_step(self, data, mask):
        """
        Perform one training step.
        
        Args:
            data: Input data with missing values replaced by random values
            mask: Binary mask (1 for observed, 0 for missing)
            
        Returns:
            Dictionary of losses
        """
        data = data.to(self.device)
        mask = mask.to(self.device)
        
        # Generate hint
        hint = self._generate_hint(mask, self.hint_rate)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Generate imputed data
        imputed_data = self.generator(data, mask)
        
        # Discriminator prediction
        d_pred = self.discriminator(imputed_data.detach(), hint)
        
        # Discriminator loss
        d_loss = self.criterion(d_pred, mask)
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate imputed data again
        imputed_data = self.generator(data, mask)
        
        # Discriminator prediction on generated data
        d_pred = self.discriminator(imputed_data, hint)
        
        # Generator loss (adversarial + reconstruction)
        g_loss_adv = -torch.mean((1 - mask) * torch.log(d_pred + 1e-8))
        g_loss_mse = torch.mean(mask * (data - imputed_data) ** 2)
        g_loss = g_loss_adv + self.alpha * g_loss_mse
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_mse': g_loss_mse.item()
        }
    
    def fit(self, data, mask, epochs=1000, batch_size=128, verbose=True):
        """
        Train GAIN model.
        
        Args:
            data: Input data (numpy array)
            mask: Binary mask (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        n_samples = data.shape[0]
        history = {'d_loss': [], 'g_loss': [], 'g_loss_adv': [], 'g_loss_mse': []}
        
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in iterator:
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_losses = {'d_loss': [], 'g_loss': [], 'g_loss_adv': [], 'g_loss_mse': []}
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]
                batch_data = torch.FloatTensor(data[batch_idx])
                batch_mask = torch.FloatTensor(mask[batch_idx])
                
                losses = self.train_step(batch_data, batch_mask)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Record average losses
            for key in history.keys():
                avg_loss = np.mean(epoch_losses[key])
                history[key].append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_description(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"D Loss: {history['d_loss'][-1]:.4f}, "
                    f"G Loss: {history['g_loss'][-1]:.4f}"
                )
        
        return history
    
    def impute(self, data, mask):
        """
        Impute missing values.
        
        Args:
            data: Input data with missing values (numpy array)
            mask: Binary mask (numpy array)
            
        Returns:
            Imputed data (numpy array)
        """
        self.generator.eval()
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            mask_tensor = torch.FloatTensor(mask).to(self.device)
            
            imputed = self.generator(data_tensor, mask_tensor)
            imputed = imputed.cpu().numpy()
        
        self.generator.train()
        
        return imputed
    
    def save_model(self, path):
        """Save model weights."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
