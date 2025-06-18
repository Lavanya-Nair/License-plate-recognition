# License Plate Recognition System

## Overview
This project implements an automated license plate recognition system using computer vision and machine learning techniques. It can detect license plates in images and recognize the characters on them.

## Project Structure
```
LicensePlateRecognition/
│
├── cca2.py              # Connected Component Analysis for plate detection
├── localization.py      # Image preprocessing and plate localization
├── segmentation.py      # Character segmentation from plates
├── prediction.py        # Main prediction pipeline
├── machine_train.py     # Model training script
├── generate_training_data.py  # Training data preparation
│
├── data/
│   ├── images/         # Training images
│   └── annotations/    # XML annotations
│
└── train/              # Generated training data
```

## Requirements
- Python 3.7+
- scikit-image
- numpy
- matplotlib
- scikit-learn
- joblib

## Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install scikit-image numpy matplotlib scikit-learn joblib
```

## Usage

### 1. Generate Training Data
```bash
python generate_training_data.py
```

### 2. Train the Model
```bash
python machine_train.py
```

### 3. Predict License Plate
```bash
python prediction.py --image path/to/image.jpg
```

## Features
- License plate detection using Connected Component Analysis
- Character segmentation from detected plates
- OCR using machine learning
- Debug visualization options
- Support for custom training data

## Debug Mode
Enable debug visualization in any module by setting `debug=True`:
```python
characters, columns = segment_characters(plate_img, debug=True)
```

## Supported Characters
- Numbers: 0-9
- Letters: A-Z (excluding I, O)

## Development
To contribute to this project:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments
- Based on scikit-image computer vision library
- Uses SVM for character classification
- Implements connected component analysis
