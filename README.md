# MLOps Assignment: Heart Disease Prediction

End-to-end ML model development with CI/CD, containerization, and production deployment.

## Project Structure

```
.
├── data/                   # Dataset and processed data
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model training and evaluation
│   ├── api/               # FastAPI application
│   └── utils/             # Utility functions
├── tests/                  # Unit tests
├── k8s/                    # Kubernetes manifests
├── screenshots/            # Deployment and pipeline screenshots
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── .github/               # GitHub Actions workflows
└── README.md              # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python src/data/download_data.py
```

### 3. Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train Model

```bash
python src/models/train.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Build Docker Image

```bash
docker build -t heart-disease-prediction:latest .
```

### 7. Run Docker Container

```bash
docker run -p 8000:8000 heart-disease-prediction:latest
```

### 8. Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

## API Usage

### Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
- Lints code
- Runs unit tests
- Trains model
- Builds Docker image
- Runs integration tests

## Monitoring

The API exposes Prometheus metrics at `/metrics`.

## Model Performance

- Model: Random Forest Classifier
- Accuracy: ~85%
- ROC-AUC: ~0.90
