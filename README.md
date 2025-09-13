# Water Segmentation using Multispectral and Optical Data

## Project Overview

This project implements a deep learning solution for accurately segmenting water bodies using multispectral and optical data. The solution is designed for monitoring water resources, flood management, and environmental conservation applications.

## Key Features

- **12-Channel Multispectral Processing**: Handles Harmonized Sentinel-2/Landsat data including:
  - Spectral bands: Coastal Aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2
  - Quality Assessment Band
  - Digital Elevation Models (Merit DEM, Copernicus DEM)
  - ESA World Cover Map
  - Water Occurrence Probability

- **U-Net Architecture**: Deep learning model with encoder-decoder structure and skip connections
- **Combined Loss Function**: BCE + Dice Loss for optimal segmentation performance
- **Comprehensive Evaluation**: IoU, Precision, Recall, F1-score metrics

## Dataset Specifications

- **Total Samples**: 200 synthetic multispectral images
- **Image Resolution**: 128×128 pixels
- **Spectral Channels**: 12 bands
- **Ground Sampling Distance**: 30m per pixel
- **Data Format**: NumPy arrays (.npy format)

## Project Structure

```
water_segmentation_project/
├── src/                    # Source code
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # U-Net architecture
│   ├── train.py           # Training utilities
│   ├── evaluate.py        # Evaluation metrics
│   └── visualize.py       # Visualization tools
├── data/                  # Dataset files
│   ├── multispectral_data.npy    # 12-channel imagery
│   └── water_masks.npy           # Binary water masks
├── models/                # Trained model checkpoints
├── results/               # Output visualizations and metrics
│   ├── multispectral_bands_showcase.png
│   ├── water_segmentation_samples.png
│   └── project_summary_comprehensive.png
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Project
```bash
python main.py
```

### Generate Dataset Only
```python
from src.data_loader import DataGenerator
generator = DataGenerator(num_samples=200)
data, masks = generator.generate_dataset()
```

### Train Model
```python
from src.model import get_model
from src.train import create_trainer

model = get_model('unet', n_channels=12, n_classes=1)
trainer = create_trainer(model, train_loader, val_loader)
history = trainer.train(num_epochs=20)
```

## Key Components

### Data Processing
- Synthetic multispectral data generation with realistic water body shapes
- Band-specific normalization for optimal model performance
- Automated train/validation/test splitting (70/10/20)

### Model Architecture
- **Input**: 12-channel multispectral data (128×128 pixels)
- **Architecture**: U-Net with 4 encoder/decoder levels
- **Parameters**: 31,042,817 trainable parameters
- **Output**: Binary water mask (128×128 pixels)

### Training Strategy
- **Loss Function**: 50% BCE Loss + 50% Dice Loss
- **Optimizer**: Adam with weight decay (1e-4)
- **Learning Rate**: 1e-4
- **Early Stopping**: Patience of 7 epochs

## Applications

This solution is suitable for:
- **Water Resource Management**: Monitor lake and river levels
- **Flood Detection**: Identify flooded areas in disaster response
- **Environmental Monitoring**: Track wetland changes over time
- **Climate Research**: Analyze water dynamics with climate patterns
- **Urban Planning**: Assess water infrastructure and flood risks
- **Agriculture**: Monitor irrigation and water stress

## Results and Visualizations

The project generates several key visualizations:

1. **Multispectral Bands Showcase**: Complete overview of all 12 spectral bands
2. **Water Segmentation Samples**: RGB composites with corresponding water masks
3. **Project Summary**: Comprehensive results and technical documentation

## Technical Specifications

- **Framework**: PyTorch for deep learning implementation
- **Data Processing**: NumPy, scikit-learn for data manipulation
- **Visualization**: Matplotlib, seaborn for result presentation
- **Image Processing**: scikit-image for morphological operations

## Model Performance Framework

The evaluation system includes:
- **IoU (Intersection over Union)**: Spatial overlap accuracy
- **Precision**: Water pixel classification accuracy
- **Recall**: Water detection completeness
- **F1-Score**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Overall classification correctness

## Future Enhancements

- Integration with real Sentinel-2/Landsat imagery
- Multi-temporal analysis capabilities
- Cloud detection and atmospheric correction
- Real-time processing pipeline
- Web-based interface for water monitoring

## Dependencies

See `requirements.txt` for complete dependency list:
- torch>=1.9.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- scikit-image>=0.18.0
- seaborn>=0.11.0
- tqdm>=4.61.0
- scipy>=1.7.0

## License

This project is developed for educational and research purposes.

## Contact

For questions, collaboration opportunities, or technical support, please reach out through the project repository.

---

**Water Segmentation Project** | Multispectral Remote Sensing | Deep Learning | September 2024
