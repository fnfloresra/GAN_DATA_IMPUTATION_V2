# GAN Data Imputation for Water Quality Monitoring

A GAN-based (Generative Adversarial Imputation Networks - GAIN) method for data imputation on multivariate time series (MTS) in the water quality domain.

## Overview

This project implements GAIN (Generative Adversarial Imputation Networks) for handling missing data in water quality monitoring time series. The method uses adversarial training to learn the distribution of observed data and generate realistic imputations for missing values.

## Features

- **GAIN Implementation**: Full implementation of Generative Adversarial Imputation Networks
- **Water Quality Focus**: Specialized for water quality parameters (pH, temperature, dissolved oxygen, etc.)
- **Multivariate Time Series**: Handles multiple correlated features simultaneously
- **Flexible Missing Mechanisms**: Supports MCAR (Missing Completely At Random) and MAR (Missing At Random)
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, MAPE) and per-feature analysis
- **Visualization Tools**: Built-in plotting for imputation results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fnfloresra/GAN_DATA_IMPUTATION_V2.git
cd GAN_DATA_IMPUTATION_V2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script for water quality data imputation:

```bash
python examples/water_quality_imputation.py
```

This will:
- Load synthetic water quality data
- Train a GAIN model
- Impute missing values
- Generate visualizations and evaluation metrics

## Usage

### Basic Usage

```python
from src.models.gan import GAIN
from src.data.water_quality import load_water_quality_data
from src.utils.preprocessing import normalize_data, create_mask, fill_missing_with_random

# Load data
data, feature_names = load_water_quality_data()

# Create mask (1 for observed, 0 for missing)
mask = (~np.isnan(data)).astype(float)

# Normalize data
norm_data, norm_params = normalize_data(data, method='minmax')

# Fill missing values with random values for training
filled_data = fill_missing_with_random(norm_data, mask)

# Initialize GAIN
gain = GAIN(
    input_dim=data.shape[1],
    hidden_dim=256,
    alpha=100,
    device='cpu'
)

# Train
history = gain.fit(filled_data, mask, epochs=500, batch_size=128)

# Impute
imputed_data = gain.impute(filled_data, mask)
```

### Custom Data

To use your own water quality data:

```python
from src.data.water_quality import load_water_quality_data

# Load from CSV file
data, feature_names = load_water_quality_data('path/to/your/data.csv')
```

Your CSV should have:
- Columns for different water quality parameters
- Optional timestamp column (will be automatically detected)
- Missing values as NaN or empty cells

## Model Architecture

### Generator
- Input: Concatenation of data and mask
- Hidden layers: 2 layers with ReLU activation
- Output: Imputed values with sigmoid activation

### Discriminator
- Input: Concatenation of imputed data and hint vector
- Hidden layers: 2 layers with ReLU activation
- Output: Prediction of which values were observed

### Loss Functions
- **Discriminator Loss**: Binary cross-entropy on mask prediction
- **Generator Loss**: Adversarial loss + weighted reconstruction loss

## Configuration

Model and training parameters can be customized in `src/config.py`:

```python
MODEL_CONFIG = {
    'hidden_dim': 256,
    'alpha': 100,           # Weight for reconstruction loss
    'hint_rate': 0.9,       # Hint rate for discriminator
    'learning_rate': 0.001,
}

TRAINING_CONFIG = {
    'epochs': 1000,
    'batch_size': 128,
    'train_ratio': 0.8,
}
```

## Project Structure

```
GAN_DATA_IMPUTATION_V2/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── gan.py              # GAIN implementation
│   ├── data/
│   │   ├── __init__.py
│   │   └── water_quality.py    # Water quality data handling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Data preprocessing utilities
│   │   └── metrics.py          # Evaluation metrics
│   ├── __init__.py
│   └── config.py               # Configuration settings
├── examples/
│   └── water_quality_imputation.py  # Example usage
├── requirements.txt
└── README.md
```

## Water Quality Parameters

The system is designed for common water quality parameters:

- **pH**: Acidity/alkalinity (0-14)
- **Temperature**: Water temperature (°C)
- **Dissolved Oxygen**: DO concentration (mg/L)
- **Conductivity**: Electrical conductivity (µS/cm)
- **Turbidity**: Water clarity (NTU)
- **Total Dissolved Solids**: TDS concentration (mg/L)
- **Chlorophyll**: Chlorophyll-a concentration (µg/L)
- **Nitrate**: Nitrate concentration (mg/L)

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Overall imputation accuracy
- **MAE** (Mean Absolute Error): Average absolute deviation
- **MAPE** (Mean Absolute Percentage Error): Relative error
- **Per-feature RMSE**: Individual feature imputation quality

## References

This implementation is based on:

- Yoon, J., Jordon, J., & Schaar, M. (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets. International Conference on Machine Learning (ICML).

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
@software{gan_data_imputation_v2,
  author = {fnfloresra},
  title = {GAN Data Imputation for Water Quality Monitoring},
  year = {2025},
  url = {https://github.com/fnfloresra/GAN_DATA_IMPUTATION_V2}
}
```
