"""
Simple usage guide for GAIN water quality data imputation.

This script demonstrates the minimal code needed to use GAIN for imputation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.models.gan import GAIN
from src.data.water_quality import load_water_quality_data
from src.utils.preprocessing import (
    normalize_data, denormalize_data, create_mask, fill_missing_with_random
)
from src.utils.metrics import calculate_rmse, calculate_mae


def simple_imputation_example():
    """
    Simple example showing the essential steps for data imputation.
    """
    print("Simple GAIN Imputation Example")
    print("=" * 50)
    
    # Step 1: Load your data
    # For real data: data, features = load_water_quality_data('path/to/data.csv')
    data, feature_names = load_water_quality_data()  # Synthetic data
    print(f"\n1. Loaded data: {data.shape}")
    
    # Step 2: Create mask (1 = observed, 0 = missing)
    mask = create_mask(data, missing_rate=0.0, mechanism='existing')
    print(f"   Missing values: {np.sum(mask == 0)} / {mask.size}")
    
    # Step 3: Normalize data
    norm_data, norm_params = normalize_data(data, method='minmax')
    
    # Step 4: Fill missing values temporarily for training
    filled_data = fill_missing_with_random(norm_data, mask)
    
    # Step 5: Initialize and train GAIN
    print("\n2. Training GAIN model...")
    gain = GAIN(
        input_dim=data.shape[1],
        hidden_dim=256,
        alpha=100,
        device='cpu'
    )
    
    gain.fit(filled_data, mask, epochs=100, batch_size=128, verbose=False)
    print("   Training complete!")
    
    # Step 6: Impute missing values
    print("\n3. Imputing missing values...")
    imputed_normalized = gain.impute(filled_data, mask)
    
    # Step 7: Denormalize to original scale
    imputed_data = denormalize_data(imputed_normalized, norm_params)
    print("   Imputation complete!")
    
    # Step 8: Use the imputed data
    print(f"\n4. Imputed data shape: {imputed_data.shape}")
    print(f"   Data range: [{imputed_data.min():.2f}, {imputed_data.max():.2f}]")
    
    # Optional: Save model for later use
    gain.save_model('/tmp/my_gain_model.pth')
    print("\n5. Model saved to /tmp/my_gain_model.pth")
    
    print("\n" + "=" * 50)
    print("✓ Imputation complete!")
    print("\nThe imputed_data array now contains your data with")
    print("missing values filled in by the GAIN model.")
    
    return imputed_data


def load_and_use_saved_model():
    """
    Example of loading a saved model and using it for imputation.
    """
    print("\n\nLoading Saved Model Example")
    print("=" * 50)
    
    # Load new data
    data, _ = load_water_quality_data()
    mask = create_mask(data, missing_rate=0.0, mechanism='existing')
    norm_data, norm_params = normalize_data(data, method='minmax')
    filled_data = fill_missing_with_random(norm_data, mask)
    
    # Load saved model
    print("\n1. Loading saved model...")
    gain = GAIN(input_dim=data.shape[1], hidden_dim=256, device='cpu')
    gain.load_model('/tmp/my_gain_model.pth')
    print("   Model loaded!")
    
    # Use it for imputation
    print("\n2. Using loaded model for imputation...")
    imputed_normalized = gain.impute(filled_data, mask)
    imputed_data = denormalize_data(imputed_normalized, norm_params)
    print("   Imputation complete!")
    
    print("\n" + "=" * 50)
    print("✓ Loaded model works!")


if __name__ == "__main__":
    # Run simple example
    imputed_data = simple_imputation_example()
    
    # Demonstrate loading saved model
    load_and_use_saved_model()
    
    print("\n\nFor more examples, see:")
    print("  - examples/test_implementation.py (unit tests)")
    print("  - examples/quick_demo.py (quick demonstration)")
    print("  - examples/water_quality_imputation.py (full example with plots)")
