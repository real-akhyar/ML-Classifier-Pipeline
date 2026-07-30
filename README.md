# ML-Classifier-Pipeline

A Machine Learning project using **Scikit-Learn** for tabular classification (Wine Quality Dataset), complete with automated model training, evaluation, artifact export, and inference.

Used for testing Roptal platform deployment & integration.

## Features
- **Dataset**: Wine dataset from `sklearn.datasets`
- **Model**: `RandomForestClassifier` with hyperparameter optimization
- **Preprocessing**: `StandardScaler`
- **Artifacts**: Exported `model.joblib` and `scaler.joblib` in `models/` directory
- **Inference**: Ready-to-use prediction script (`src/predict.py`)

## Structure
```
ML-Classifier-Pipeline/
├── models/             # Saved model & scaler binaries (.joblib)
├── src/
│   ├── train.py        # Model training and export script
│   └── predict.py      # Model inference script
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Export Model
```bash
python src/train.py
```

### 3. Run Inference
```bash
python src/predict.py
```
