"""Water quality dataset handling utilities."""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class WaterQualityDataset(Dataset):
    """PyTorch Dataset for water quality time series data."""
    
    def __init__(self, data, mask):
        """
        Initialize dataset.
        
        Args:
            data: Time series data (numpy array)
            mask: Binary mask (numpy array)
        """
        self.data = data
        self.mask = mask
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]


def load_water_quality_data(filepath=None, return_dataframe=False, 
                            synthetic_missing_rate=0.07):
    """
    Load water quality time series data.
    
    If no filepath is provided, generates synthetic water quality data for demonstration.
    
    Args:
        filepath: Path to CSV file (optional)
        return_dataframe: Whether to return pandas DataFrame
        synthetic_missing_rate: Missing rate for synthetic data (default: 0.07)
        
    Returns:
        data: Time series data
        feature_names: List of feature names
        dataframe: Original dataframe (if return_dataframe=True)
    """
    if filepath is not None:
        # Load from CSV
        df = pd.read_csv(filepath)
        
        # Assume first column might be timestamp
        if df.columns[0].lower() in ['date', 'time', 'timestamp', 'datetime']:
            df = df.set_index(df.columns[0])
        
        feature_names = df.columns.tolist()
        data = df.values
        
        if return_dataframe:
            return data, feature_names, df
        else:
            return data, feature_names
    
    else:
        # Generate synthetic water quality data
        n_samples = 1000
        n_features = 8
        
        # Feature names typical for water quality monitoring
        feature_names = [
            'pH',
            'Temperature',
            'Dissolved_Oxygen',
            'Conductivity',
            'Turbidity',
            'Total_Dissolved_Solids',
            'Chlorophyll',
            'Nitrate'
        ]
        
        # Generate synthetic time series data with realistic patterns
        time = np.linspace(0, 4 * np.pi, n_samples)
        
        data = np.zeros((n_samples, n_features))
        
        # pH: typically 6.5-8.5, with daily variations
        data[:, 0] = 7.5 + 0.5 * np.sin(time) + 0.2 * np.random.randn(n_samples)
        
        # Temperature: seasonal pattern
        data[:, 1] = 20 + 10 * np.sin(time / 2) + 2 * np.random.randn(n_samples)
        
        # Dissolved Oxygen: inversely related to temperature
        data[:, 2] = 8 - 2 * np.sin(time / 2) + 0.5 * np.random.randn(n_samples)
        
        # Conductivity: relatively stable with small variations
        data[:, 3] = 500 + 50 * np.sin(time / 3) + 20 * np.random.randn(n_samples)
        
        # Turbidity: random spikes (events)
        data[:, 4] = 10 + 5 * np.abs(np.random.randn(n_samples))
        spike_indices = np.random.choice(n_samples, size=50, replace=False)
        data[spike_indices, 4] += 20 * np.random.rand(50)
        
        # Total Dissolved Solids: correlated with conductivity
        data[:, 5] = 0.65 * data[:, 3] + 20 * np.random.randn(n_samples)
        
        # Chlorophyll: seasonal pattern (algae growth)
        data[:, 6] = 5 + 3 * np.sin(time / 2 + np.pi / 4) + 1 * np.random.randn(n_samples)
        data[:, 6] = np.maximum(data[:, 6], 0)  # Non-negative
        
        # Nitrate: periodic pattern
        data[:, 7] = 2 + 1.5 * np.sin(time / 4) + 0.5 * np.random.randn(n_samples)
        data[:, 7] = np.maximum(data[:, 7], 0)  # Non-negative
        
        # Add some missing values randomly
        missing_mask = np.random.rand(n_samples, n_features) < synthetic_missing_rate
        data[missing_mask] = np.nan
        
        if return_dataframe:
            df = pd.DataFrame(data, columns=feature_names)
            return data, feature_names, df
        else:
            return data, feature_names


def create_sequences(data, mask, sequence_length=10, stride=1):
    """
    Create sequences from time series data for sequence-based models.
    
    Args:
        data: Time series data (numpy array)
        mask: Binary mask (numpy array)
        sequence_length: Length of each sequence
        stride: Step size between sequences
        
    Returns:
        sequences: Array of sequences
        sequence_masks: Array of sequence masks
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    sequences = []
    sequence_masks = []
    
    for i in range(0, n_samples - sequence_length + 1, stride):
        seq = data[i:i + sequence_length]
        seq_mask = mask[i:i + sequence_length]
        
        sequences.append(seq)
        sequence_masks.append(seq_mask)
    
    sequences = np.array(sequences)
    sequence_masks = np.array(sequence_masks)
    
    return sequences, sequence_masks


def add_temporal_features(data, feature_names):
    """
    Add temporal features to the dataset.
    
    Args:
        data: Time series data (numpy array)
        feature_names: List of feature names
        
    Returns:
        Enhanced data with temporal features
        Updated feature names
    """
    n_samples = data.shape[0]
    
    # Add time index normalized
    time_index = np.arange(n_samples).reshape(-1, 1) / n_samples
    
    # Add sine and cosine of time for periodicity
    time_sin = np.sin(2 * np.pi * time_index)
    time_cos = np.cos(2 * np.pi * time_index)
    
    # Concatenate temporal features
    enhanced_data = np.concatenate([data, time_index, time_sin, time_cos], axis=1)
    
    enhanced_feature_names = feature_names + ['time_index', 'time_sin', 'time_cos']
    
    return enhanced_data, enhanced_feature_names
