"""
Example usage of GAIN for water quality data imputation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models.gan import GAIN
from src.data.water_quality import load_water_quality_data
from src.utils.preprocessing import (
    normalize_data, denormalize_data, create_mask, 
    fill_missing_with_random, temporal_split
)
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_feature_rmse


def plot_imputation_results(original_data, imputed_data, mask, feature_names, 
                            sample_range=(0, 200), save_path=None):
    """
    Plot original vs imputed data for visualization.
    
    Args:
        original_data: Original data with missing values
        imputed_data: Imputed data
        mask: Binary mask
        feature_names: List of feature names
        sample_range: Range of samples to plot
        save_path: Path to save the figure
    """
    n_features = min(4, len(feature_names))  # Plot first 4 features
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
    
    if n_features == 1:
        axes = [axes]
    
    start, end = sample_range
    x = np.arange(start, end)
    
    for i in range(n_features):
        ax = axes[i]
        
        # Plot observed values
        observed_idx = mask[start:end, i] == 1
        ax.plot(x[observed_idx], original_data[start:end, i][observed_idx], 
                'o', label='Observed', markersize=3, alpha=0.6)
        
        # Plot imputed values
        missing_idx = mask[start:end, i] == 0
        if np.any(missing_idx):
            ax.plot(x[missing_idx], imputed_data[start:end, i][missing_idx], 
                    'x', label='Imputed', markersize=5, color='red')
        
        # Plot full imputed series as line
        ax.plot(x, imputed_data[start:end, i], '-', alpha=0.3, color='gray', 
                label='Imputed series')
        
        ax.set_ylabel(feature_names[i])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to demonstrate GAIN imputation."""
    
    print("=" * 60)
    print("GAIN: Water Quality Data Imputation")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load water quality data
    print("\nLoading water quality data...")
    data, feature_names = load_water_quality_data()
    print(f"Data shape: {data.shape}")
    print(f"Features: {feature_names}")
    
    # Keep a copy of original data with natural missing values
    original_data = data.copy()
    original_mask = (~np.isnan(data)).astype(float)
    
    print(f"Original missing rate: {1 - np.mean(original_mask):.2%}")
    
    # For evaluation, we'll artificially introduce more missing values
    # on complete data and measure imputation quality
    complete_data = data.copy()
    
    # Fill existing missing values with forward fill for evaluation
    for j in range(complete_data.shape[1]):
        col = complete_data[:, j]
        mask_col = ~np.isnan(col)
        if np.any(mask_col):
            # Forward fill
            last_valid = None
            for i in range(len(col)):
                if not np.isnan(col[i]):
                    last_valid = col[i]
                elif last_valid is not None:
                    col[i] = last_valid
            # Backward fill for any remaining NaN at the start
            last_valid = None
            for i in range(len(col) - 1, -1, -1):
                if not np.isnan(col[i]):
                    last_valid = col[i]
                elif last_valid is not None:
                    col[i] = last_valid
    
    # Create artificial missing values for evaluation
    artificial_missing_rate = 0.2
    evaluation_mask = create_mask(complete_data, 
                                  missing_rate=artificial_missing_rate, 
                                  mechanism='MCAR')
    
    print(f"Artificial missing rate for evaluation: {1 - np.mean(evaluation_mask):.2%}")
    
    # Apply mask to data
    masked_data = complete_data.copy()
    masked_data[evaluation_mask == 0] = np.nan
    
    # Normalize data
    print("\nNormalizing data...")
    norm_data, norm_params = normalize_data(masked_data, method='minmax')
    
    # Fill missing values with random values for GAN input
    filled_data = fill_missing_with_random(norm_data, evaluation_mask)
    
    # Split data
    train_data, train_mask, test_data, test_mask = temporal_split(
        filled_data, evaluation_mask, train_ratio=0.8
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize GAIN
    print("\nInitializing GAIN model...")
    input_dim = data.shape[1]
    gain = GAIN(
        input_dim=input_dim,
        hidden_dim=256,
        alpha=100,
        hint_rate=0.9,
        learning_rate=0.001,
        device=device
    )
    
    # Train GAIN
    print("\nTraining GAIN...")
    history = gain.fit(
        train_data,
        train_mask,
        epochs=500,
        batch_size=128,
        verbose=True
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved to training_history.png")
    plt.close()
    
    # Impute test data
    print("\nImputing test data...")
    imputed_test_normalized = gain.impute(test_data, test_mask)
    
    # Denormalize
    imputed_test = denormalize_data(imputed_test_normalized, norm_params)
    test_data_denorm = denormalize_data(test_data, norm_params)
    
    # Get ground truth for test data
    test_ground_truth = complete_data[-len(test_data):]
    
    # Calculate metrics
    print("\nEvaluation Metrics (Test Set):")
    print("-" * 40)
    
    rmse = calculate_rmse(test_ground_truth, imputed_test, test_mask)
    mae = calculate_mae(test_ground_truth, imputed_test, test_mask)
    
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    
    print("\nPer-feature RMSE:")
    feature_rmse = calculate_feature_rmse(test_ground_truth, imputed_test, test_mask)
    for feature, rmse_val in feature_rmse.items():
        feature_idx = int(feature.split('_')[1])
        print(f"  {feature_names[feature_idx]}: {rmse_val:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    plot_imputation_results(
        test_ground_truth,
        imputed_test,
        test_mask,
        feature_names,
        sample_range=(0, min(200, len(test_data))),
        save_path='imputation_results.png'
    )
    
    # Save model
    print("\nSaving model...")
    gain.save_model('gain_model.pth')
    print("Model saved to gain_model.pth")
    
    print("\n" + "=" * 60)
    print("Imputation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
