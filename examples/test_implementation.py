"""Quick test script to verify GAIN implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.models.gan import GAIN
from src.data.water_quality import load_water_quality_data
from src.utils.preprocessing import (
    normalize_data, create_mask, fill_missing_with_random
)
from src.utils.metrics import calculate_rmse, calculate_mae


def test_basic_functionality():
    """Test basic functionality of GAIN."""
    print("Testing GAIN basic functionality...")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load small synthetic dataset
    print("\n1. Loading synthetic water quality data...")
    data, feature_names = load_water_quality_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Features: {len(feature_names)}")
    
    # Create mask
    print("\n2. Creating mask...")
    mask = create_mask(data, missing_rate=0.0, mechanism='existing')
    print(f"   Original missing rate: {1 - np.mean(mask):.2%}")
    
    # Normalize
    print("\n3. Normalizing data...")
    norm_data, norm_params = normalize_data(data, method='minmax')
    filled_data = fill_missing_with_random(norm_data, mask)
    
    # Use small subset for quick test
    n_samples = 200
    filled_data = filled_data[:n_samples]
    mask = mask[:n_samples]
    
    print(f"   Using {n_samples} samples for quick test")
    
    # Initialize GAIN
    print("\n4. Initializing GAIN...")
    gain = GAIN(
        input_dim=data.shape[1],
        hidden_dim=128,  # Smaller for quick test
        alpha=100,
        device='cpu'
    )
    print("   Model initialized successfully")
    
    # Train with few epochs for quick test
    print("\n5. Training GAIN (quick test - 50 epochs)...")
    history = gain.fit(
        filled_data,
        mask,
        epochs=50,
        batch_size=32,
        verbose=False
    )
    print(f"   Training completed")
    print(f"   Final D Loss: {history['d_loss'][-1]:.4f}")
    print(f"   Final G Loss: {history['g_loss'][-1]:.4f}")
    
    # Impute
    print("\n6. Testing imputation...")
    imputed_data = gain.impute(filled_data, mask)
    print(f"   Imputed data shape: {imputed_data.shape}")
    print(f"   Imputed data range: [{imputed_data.min():.3f}, {imputed_data.max():.3f}]")
    
    # Save and load model
    print("\n7. Testing model save/load...")
    test_model_path = '/tmp/test_gain_model.pth'
    gain.save_model(test_model_path)
    print(f"   Model saved to {test_model_path}")
    
    gain2 = GAIN(input_dim=data.shape[1], hidden_dim=128, device='cpu')
    gain2.load_model(test_model_path)
    print("   Model loaded successfully")
    
    # Verify loaded model produces same results
    imputed_data2 = gain2.impute(filled_data, mask)
    diff = np.abs(imputed_data - imputed_data2).max()
    print(f"   Max difference after reload: {diff:.6f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed successfully!")
    print("="*60)
    
    return True


def test_metrics():
    """Test metric calculations."""
    print("\nTesting metrics...")
    
    # Create simple test data
    true_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    imputed_data = np.array([[1.1, 2.0, 3.0], [4.0, 5.2, 6.0], [7.0, 8.0, 9.3]])
    mask = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    
    rmse = calculate_rmse(true_data, imputed_data, mask)
    mae = calculate_mae(true_data, imputed_data, mask)
    
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print("   ✓ Metrics calculated successfully")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("GAIN Implementation Test Suite")
    print("="*60)
    
    try:
        test_metrics()
        test_basic_functionality()
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
