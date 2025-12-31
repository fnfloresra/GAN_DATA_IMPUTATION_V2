"""Evaluation metrics for imputation quality."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse(true_data, imputed_data, mask):
    """
    Calculate Root Mean Squared Error for imputed values.
    
    Args:
        true_data: True values (numpy array)
        imputed_data: Imputed values (numpy array)
        mask: Binary mask (1 for observed, 0 for missing/imputed)
        
    Returns:
        RMSE value
    """
    # Only evaluate on missing values
    missing_mask = (mask == 0)
    
    if np.sum(missing_mask) == 0:
        return 0.0
    
    true_missing = true_data[missing_mask]
    imputed_missing = imputed_data[missing_mask]
    
    rmse = np.sqrt(mean_squared_error(true_missing, imputed_missing))
    return rmse


def calculate_mae(true_data, imputed_data, mask):
    """
    Calculate Mean Absolute Error for imputed values.
    
    Args:
        true_data: True values (numpy array)
        imputed_data: Imputed values (numpy array)
        mask: Binary mask (1 for observed, 0 for missing/imputed)
        
    Returns:
        MAE value
    """
    # Only evaluate on missing values
    missing_mask = (mask == 0)
    
    if np.sum(missing_mask) == 0:
        return 0.0
    
    true_missing = true_data[missing_mask]
    imputed_missing = imputed_data[missing_mask]
    
    mae = mean_absolute_error(true_missing, imputed_missing)
    return mae


def calculate_mape(true_data, imputed_data, mask, epsilon=1e-8):
    """
    Calculate Mean Absolute Percentage Error for imputed values.
    
    Args:
        true_data: True values (numpy array)
        imputed_data: Imputed values (numpy array)
        mask: Binary mask (1 for observed, 0 for missing/imputed)
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (percentage)
    """
    # Only evaluate on missing values
    missing_mask = (mask == 0)
    
    if np.sum(missing_mask) == 0:
        return 0.0
    
    true_missing = true_data[missing_mask]
    imputed_missing = imputed_data[missing_mask]
    
    # Avoid division by zero
    mape = np.mean(np.abs((true_missing - imputed_missing) / (true_missing + epsilon))) * 100
    return mape


def calculate_feature_rmse(true_data, imputed_data, mask):
    """
    Calculate RMSE for each feature separately.
    
    Args:
        true_data: True values (numpy array)
        imputed_data: Imputed values (numpy array)
        mask: Binary mask (1 for observed, 0 for missing/imputed)
        
    Returns:
        Dictionary with RMSE for each feature
    """
    n_features = true_data.shape[1]
    feature_rmse = {}
    
    for i in range(n_features):
        feature_mask = mask[:, i]
        missing_mask = (feature_mask == 0)
        
        if np.sum(missing_mask) > 0:
            true_missing = true_data[missing_mask, i]
            imputed_missing = imputed_data[missing_mask, i]
            rmse = np.sqrt(mean_squared_error(true_missing, imputed_missing))
            feature_rmse[f'feature_{i}'] = rmse
        else:
            feature_rmse[f'feature_{i}'] = 0.0
    
    return feature_rmse
