"""
Quick demo of GAIN for water quality data imputation with reduced training time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch

from src.models.gan import GAIN
from src.data.water_quality import load_water_quality_data
from src.utils.preprocessing import (
    normalize_data, denormalize_data, create_mask, 
    fill_missing_with_random, temporal_split
)
from src.utils.metrics import calculate_rmse, calculate_mae, calculate_feature_rmse


def main():
    """Main function to demonstrate GAIN imputation."""
    
    print("=" * 60)
    print("GAIN: Water Quality Data Imputation Demo")
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
    
    # Keep a copy for evaluation
    complete_data = data.copy()
    
    # Fill existing missing values for complete data
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
            # Backward fill
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
    
    print(f"\nArtificial missing rate: {1 - np.mean(evaluation_mask):.2%}")
    
    # Apply mask
    masked_data = complete_data.copy()
    masked_data[evaluation_mask == 0] = np.nan
    
    # Normalize
    print("\nNormalizing data...")
    norm_data, norm_params = normalize_data(masked_data, method='minmax')
    filled_data = fill_missing_with_random(norm_data, evaluation_mask)
    
    # Split data
    train_data, train_mask, test_data, test_mask = temporal_split(
        filled_data, evaluation_mask, train_ratio=0.8
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize GAIN
    print("\nInitializing GAIN model...")
    gain = GAIN(
        input_dim=data.shape[1],
        hidden_dim=256,
        alpha=100,
        hint_rate=0.9,
        learning_rate=0.001,
        device=device
    )
    
    # Train GAIN (reduced epochs for demo)
    print("\nTraining GAIN (100 epochs for demo)...")
    history = gain.fit(
        train_data,
        train_mask,
        epochs=100,
        batch_size=128,
        verbose=True
    )
    
    # Impute test data
    print("\nImputing test data...")
    imputed_test_normalized = gain.impute(test_data, test_mask)
    
    # Denormalize
    imputed_test = denormalize_data(imputed_test_normalized, norm_params)
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
    
    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
