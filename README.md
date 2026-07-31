# ML-Classifier-Pipeline

A production-ready Machine Learning deployment package using **Scikit-Learn** and **FastAPI** for tabular classification (Wine Quality Dataset), complete with Docker containerization, REST API, automated model training, artifact export, and inference.

Configured and ready for instant deployment on **Roptal** ([www.roptal.com](https://www.roptal.com) / [www.oryvoai.com](https://www.oryvoai.com)).

## Features
- **ML Framework**: Scikit-Learn (`RandomForestClassifier`, `StandardScaler`)
- **API Framework**: FastAPI + Uvicorn
- **Deployment Ready**: `Dockerfile` + `docker-compose.yml`
- **Artifacts**: Pre-trained `model.joblib` and `scaler.joblib` in `models/` directory
- **Endpoints**: Interactive OpenAPI docs (`/docs`), health check (`/health`), predictions (`/predict`)

## Project Structure
```
ML-Classifier-Pipeline/
├── app.py              # FastAPI REST API Server
├── Dockerfile          # Production Docker Container Specification
├── docker-compose.yml  # Docker Compose Configuration
├── .dockerignore       # Docker Build Exclusion Rules
├── models/             # Exported model & scaler binaries (.joblib)
├── src/
│   ├── train.py        # Model training and export script
│   └── predict.py      # Standalone inference script
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Quick Start (Docker Deployment)

### 1. Build and Run via Docker Compose
```bash
docker-compose up --build -d
```
The service will be live at `http://localhost:8000`.

### 2. Build and Run via Docker CLI
```bash
# Build Docker image
docker build -t ml-classifier-pipeline .

# Run Docker container
docker run -d -p 8000:8000 --name ml-classifier ml-classifier-pipeline
```

---

## API Endpoints

### `GET /health`
Returns the status of the loaded model.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "num_features_expected": 13
}
```

### `POST /predict`
Submit feature values for classification.
```json
{
  "features": [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
}
```
**Response**:
```json
{
  "status": "success",
  "class_index": 0,
  "class_name": "class_0",
  "probabilities": {
    "class_0": 0.9874,
    "class_1": 0.0126,
    "class_2": 0.0
  }
}
```

---

## Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Start local server
uvicorn app:app --reload --port 8000
```
