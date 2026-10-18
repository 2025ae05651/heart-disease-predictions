# Quick Start Guide - MLOps Assignment

## Project Overview
This is a complete end-to-end MLOps project for heart disease prediction with CI/CD, containerization, and production deployment.

## Project Structure
```
MlOps Assignment/
├── data/                   # Dataset directory
├── notebooks/              # EDA scripts
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model training
│   ├── api/               # FastAPI application
│   └── utils/             # Utilities
├── tests/                  # Unit tests
├── k8s/                    # Kubernetes manifests
├── screenshots/            # Visualizations
├── models/                 # Trained models
├── .github/               # GitHub Actions workflows
├── requirements.txt        # Dependencies
├── Dockerfile             # Container definition
├── README.md              # Project documentation
└── REPORT.md              # Assignment report
```

## Next Steps

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download and Preprocess Data
```bash
python src/data/download_data.py
python src/data/preprocess.py
```

### 3. Run EDA
```bash
python notebooks/01_eda.py
```

### 4. Train Models
```bash
python src/models/train.py
```

### 5. Run Tests
```bash
pytest tests/ -v
```

### 6. Test API Locally
```bash
python -m uvicorn src.api.app:app --reload
```

### 7. Build Docker Image
```bash
docker build -t heart-disease-prediction:latest .
docker run -p 8000:8000 heart-disease-prediction:latest
```

### 8. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/monitoring.yaml
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction
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

## GitHub Setup

1. Initialize Git repository:
```bash
git init
git add .
git commit -m "Initial commit - MLOps Assignment"
```

2. Create repository on GitHub and push:
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

3. The CI/CD pipeline will automatically run on push.

## Screenshots Needed for Report

After running the pipeline, capture screenshots of:
1. EDA visualizations (automatically saved in screenshots/)
2. MLflow experiment tracking UI
3. GitHub Actions workflow runs
4. Docker container running
5. Kubernetes pods and services
6. Grafana dashboard
7. API predictions

## Report Customization

Edit `REPORT.md` to add:
- Your name and student ID
- Actual screenshot links
- Repository URL
- Video demonstration link
- Any personal observations

## Important Notes

- The project uses the UCI Heart Disease dataset
- Random Forest is selected as the best model (ROC-AUC: ~0.90)
- All code follows production best practices
- The CI/CD pipeline includes linting, testing, training, and building
- Kubernetes deployment includes 3 replicas for high availability
- Monitoring is set up with Prometheus and Grafana

## Troubleshooting

### MLflow not starting
```bash
mlflow ui
```
Then access at http://localhost:5000

### Docker build fails
Ensure Docker is running and you have sufficient disk space.

### Kubernetes deployment fails
Check if Minikube/Docker Desktop is running:
```bash
kubectl get nodes
```

### Tests failing
Ensure data has been downloaded and preprocessed first.

## Academic Integrity

This is a template project. You should:
- Run the entire pipeline yourself
- Generate your own screenshots
- Add your own analysis and observations to the report
- Consider making improvements or modifications to demonstrate originality
- Document any changes you make

## Support

For issues or questions, refer to:
- Assignment instructions PDF
- Project README.md
- Documentation links in REPORT.md
