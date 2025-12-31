"""Preprocessing utilities for time series data."""

import numpy as np


def normalize_data(data, method='minmax'):
    """
    Normalize data.
    
    Args:
        data: Input data (numpy array)
        method: Normalization method ('minmax' or 'standard')
        
    Returns:
        Normalized data and normalization parameters
    """
    if method == 'minmax':
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        
        normalized = (data - min_val) / range_val
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'standard':
        mean_val = np.nanmean(data, axis=0)
        std_val = np.nanstd(data, axis=0)
        
        # Avoid division by zero
        std_val[std_val == 0] = 1
        
        normalized = (data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val, 'method': 'standard'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data, params):
    """
    Denormalize data.
    
    Args:
        data: Normalized data (numpy array)
        params: Normalization parameters
        
    Returns:
        Denormalized data
    """
    if params['method'] == 'minmax':
        denormalized = data * (params['max'] - params['min']) + params['min']
    elif params['method'] == 'standard':
        denormalized = data * params['std'] + params['mean']
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")
    
    return denormalized


def create_mask(data, missing_rate=0.0, mechanism='MCAR'):
    """
    Create binary mask for missing data.
    
    Args:
        data: Input data (numpy array)
        missing_rate: Additional artificial missing rate for testing
        mechanism: Missing mechanism ('MCAR', 'MAR', or 'existing')
        
    Returns:
        Binary mask (1 for observed, 0 for missing)
    """
    n_samples, n_features = data.shape
    
    # Start with existing missing values
    mask = (~np.isnan(data)).astype(float)
    
    if missing_rate > 0 and mechanism != 'existing':
        if mechanism == 'MCAR':
            # Missing Completely At Random
            additional_missing = np.random.rand(n_samples, n_features) > missing_rate
            mask = mask * additional_missing.astype(float)
            
        elif mechanism == 'MAR':
            # Missing At Random (simplified version)
            # Higher probability of missing for values below median
            for j in range(n_features):
                feature_data = data[:, j]
                median_val = np.nanmedian(feature_data)
                prob = np.where(feature_data < median_val, 
                               missing_rate * 1.5, 
                               missing_rate * 0.5)
                additional_missing = np.random.rand(n_samples) > prob
                mask[:, j] = mask[:, j] * additional_missing.astype(float)
        
        else:
            raise ValueError(f"Unknown missing mechanism: {mechanism}")
    
    return mask


def fill_missing_with_random(data, mask):
    """
    Fill missing values with random values from observed data.
    
    Args:
        data: Input data with missing values
        mask: Binary mask
        
    Returns:
        Data with missing values filled
    """
    filled_data = data.copy()
    
    for j in range(data.shape[1]):
        # Get observed values for this feature
        observed_values = data[mask[:, j] == 1, j]
        
        if len(observed_values) > 0:
            # Fill missing values with random samples from observed
            missing_indices = mask[:, j] == 0
            n_missing = np.sum(missing_indices)
            
            if n_missing > 0:
                random_values = np.random.choice(observed_values, n_missing, replace=True)
                filled_data[missing_indices, j] = random_values
        else:
            # If no observed values, fill with 0
            filled_data[mask[:, j] == 0, j] = 0
    
    return filled_data


def temporal_split(data, mask, train_ratio=0.8):
    """
    Split time series data temporally.
    
    Args:
        data: Input data
        mask: Binary mask
        train_ratio: Ratio of training data
        
    Returns:
        Training and test data with masks
    """
    n_samples = data.shape[0]
    split_idx = int(n_samples * train_ratio)
    
    train_data = data[:split_idx]
    train_mask = mask[:split_idx]
    test_data = data[split_idx:]
    test_mask = mask[split_idx:]
    
    return train_data, train_mask, test_data, test_mask
