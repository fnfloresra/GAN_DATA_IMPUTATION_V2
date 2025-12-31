"""Configuration settings for GAIN model."""

# Model hyperparameters
MODEL_CONFIG = {
    'hidden_dim': 256,          # Hidden layer dimension
    'alpha': 100,                # Weight for reconstruction loss
    'hint_rate': 0.9,            # Hint rate for discriminator
    'learning_rate': 0.001,      # Learning rate for optimizers
}

# Training parameters
TRAINING_CONFIG = {
    'epochs': 1000,              # Number of training epochs
    'batch_size': 128,           # Batch size for training
    'train_ratio': 0.8,          # Ratio of training data
}

# Data preprocessing
DATA_CONFIG = {
    'normalization': 'minmax',   # Normalization method ('minmax' or 'standard')
    'missing_rate': 0.2,         # Artificial missing rate for testing
    'missing_mechanism': 'MCAR', # Missing mechanism ('MCAR', 'MAR', or 'existing')
}

# Water quality features
WATER_QUALITY_FEATURES = [
    'pH',
    'Temperature',
    'Dissolved_Oxygen',
    'Conductivity',
    'Turbidity',
    'Total_Dissolved_Solids',
    'Chlorophyll',
    'Nitrate'
]

# Feature ranges (for validation)
FEATURE_RANGES = {
    'pH': (0, 14),
    'Temperature': (0, 40),
    'Dissolved_Oxygen': (0, 20),
    'Conductivity': (0, 2000),
    'Turbidity': (0, 100),
    'Total_Dissolved_Solids': (0, 1000),
    'Chlorophyll': (0, 50),
    'Nitrate': (0, 20)
}
