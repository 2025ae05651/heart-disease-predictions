# Complete Fixes Guide for MLOps Assignment

This guide will help you fix all the identified issues in the project.

## ✅ Code Fixes (Already Applied)

The following code fixes have already been applied to the final version:
- ✅ Fixed preprocessor to save scaler and feature columns separately
- ✅ Fixed API to load scaler and feature columns instead of preprocessor class
- ✅ Fixed Pydantic V2 configuration (json_schema_extra)

## 📋 Step-by-Step Instructions to Fix Remaining Issues

### Step 1: Generate Model Files and Dataset

Navigate to the project directory and run:

```powershell
cd "C:\Users\bosea2\Downloads\heart-disease-prediction-main"

# Activate virtual environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/data/download_data.py

# Preprocess data (this will create scaler.joblib and feature_columns.joblib)
python src/data/preprocess.py

# Run EDA to generate visualizations
python notebooks/01_eda.py

# Train models (this will create model files)
python src/models/train.py

# Run tests
python -m pytest tests/ -v
```

**Expected Output:**
- `data/heart_disease_raw.csv` - Raw dataset
- `data/scaler.joblib` - Fitted scaler
- `data/feature_columns.joblib` - Feature column names
- `models/best_model.joblib` - Best trained model
- `models/logistic_regression.joblib` - LR model
- `models/random_forest.joblib` - RF model
- `screenshots/*.png` - All EDA visualizations

---

### Step 2: Test API Locally

```powershell
# Start API server
python -m uvicorn src.api.app:app --reload
```

In a **new PowerShell terminal**:

```powershell
cd "C:\Users\bosea2\Downloads\heart-disease-prediction-main"
venv\Scripts\activate

# Test health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# Test prediction
$body = @{
    age = 52
    sex = 1
    cp = 0
    trestbps = 125
    chol = 212
    fbs = 0
    restecg = 1
    thalach = 168
    exang = 0
    oldpeak = 1.0
    slope = 2
    ca = 2
    thal = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"

# Test metrics
Invoke-RestMethod -Uri "http://localhost:8000/metrics" -Method Get
```

**Screenshot Required:** Capture the API response showing successful prediction.

---

### Step 3: Build and Test Docker Image

```powershell
# Build Docker image
docker build -t heart-disease-prediction:latest .

# Run Docker container
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-prediction:latest

# Wait 10 seconds for container to start
Start-Sleep -Seconds 10

# Test the container
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# Stop and remove container
docker stop heart-disease-api
docker rm heart-disease-api
```

**Screenshot Required:** Capture the Docker build output showing successful build.

---

### Step 4: Deploy to Kubernetes (Local)

#### Option A: Using Docker Desktop (Recommended)

1. Enable Kubernetes in Docker Desktop:
   - Open Docker Desktop
   - Go to Settings → Kubernetes
   - Enable Kubernetes
   - Click "Apply & Restart"

2. Deploy application:

```powershell
# Check Kubernetes is running
kubectl get nodes

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check pods
kubectl get pods

# Check services
kubectl get services

# Get service URL
kubectl get service heart-disease-prediction-service
```

3. Test the deployment:

```powershell
# Get the service URL (use NodePort if LoadBalancer doesn't work)
$service = kubectl get service heart-disease-prediction-service -o json
$port = $service.spec.ports[0].nodePort
$nodeIP = kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}'

# Test the service
Invoke-RestMethod -Uri "http://${nodeIP}:${port}/health" -Method Get
```

#### Option B: Using Minikube

```powershell
# Start Minikube
minikube start

# Enable ingress
minikube addons enable ingress

# Build image for Minikube
eval $(minikube docker-env)
docker build -t heart-disease-prediction:latest .

# Deploy
kubectl apply -f k8s/deployment.yaml

# Get URL
minikube service heart-disease-prediction-service --url
```

**Screenshots Required:**
1. `kubectl get pods` showing running pods
2. `kubectl get services` showing service
3. API response from Kubernetes deployment

---

### Step 5: Deploy Monitoring Stack

```powershell
# Apply monitoring manifests
kubectl apply -f k8s/monitoring.yaml

# Wait for pods to be ready
kubectl get pods -w

# Get Grafana URL
kubectl get service grafana-service

# Get Prometheus URL
kubectl get service prometheus-service
```

**Access Grafana:**
1. Get Grafana URL: `kubectl get service grafana-service`
2. Open in browser (use NodePort if LoadBalancer doesn't work)
3. Login with: admin / admin
4. Add Prometheus as data source:
   - URL: `http://prometheus-service:9090`
5. Import dashboard or create custom panels

**Screenshots Required:**
1. Grafana dashboard showing API metrics
2. Prometheus metrics page
3. Kubernetes pods showing monitoring stack

---

### Step 6: Capture All Screenshots

Create a folder structure for screenshots:

```powershell
# Create screenshot folders
New-Item -ItemType Directory -Force -Path screenshots\deployment
New-Item -ItemType Directory -Force -Path screenshots\monitoring
New-Item -ItemType Directory -Force -Path screenshots\api
New-Item -ItemType Directory -Force -Path screenshots\docker
New-Item -ItemType Directory -Force -Path screenshots\cicd
```

**Required Screenshots:**

1. **EDA Screenshots** (already in screenshots/):
   - ✅ 01_target_distribution.png
   - ✅ 02_feature_distributions.png
   - ✅ 03_correlation_heatmap.png
   - ✅ 04_features_by_target.png
   - ✅ 05_lr_confusion_matrix.png
   - ✅ 06_rf_confusion_matrix.png

2. **API Screenshots** (screenshots/api/):
   - API health check response
   - API prediction response
   - API metrics endpoint

3. **Docker Screenshots** (screenshots/docker/):
   - Docker build output
   - Docker container running
   - Docker container test response

4. **Kubernetes Screenshots** (screenshots/deployment/):
   - `kubectl get pods` output
   - `kubectl get services` output
   - `kubectl describe deployment` output
   - API response from Kubernetes

5. **Monitoring Screenshots** (screenshots/monitoring/):
   - Grafana dashboard with metrics
   - Prometheus UI
   - Kubernetes monitoring pods

6. **CI/CD Screenshots** (screenshots/cicd/):
   - GitHub Actions workflow runs
   - Pipeline status
   - Test results

---

### Step 7: Setup GitHub Repository

```powershell
# Initialize git (if not already done)
cd "C:\Users\bosea2\Downloads\heart-disease-prediction-main"
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - MLOps Assignment"

# Create repository on GitHub (go to github.com and create new repo)

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git

# Push
git branch -M main
git push -u origin main
```

**Note:** The CI/CD pipeline will automatically run after pushing.

**Screenshot Required:** GitHub repository page with code.

---

### Step 8: Record Video Demonstration

**Video Content (2-5 minutes):**

1. **Introduction** (30 seconds):
   - Show project structure
   - Explain the problem statement

2. **Data Pipeline** (1 minute):
   - Show data download
   - Show EDA visualizations
   - Show preprocessing

3. **Model Training** (1 minute):
   - Show training script running
   - Show MLflow experiment tracking (if running `mlflow ui`)
   - Show model comparison

4. **API Testing** (30 seconds):
   - Show API running
   - Show prediction request/response

5. **Deployment** (1 minute):
   - Show Docker build
   - Show Kubernetes deployment
   - Show monitoring dashboard

6. **Conclusion** (30 seconds):
   - Summary of achievements
   - Show GitHub repository

**Recording Tools:**
- Windows: Xbox Game Bar (Win + G)
- OBS Studio (free)
- Microsoft Teams recording

**Save as:** `demonstration.mp4` in project root

---

### Step 9: Create Final Report (Word/PDF)

**Convert REPORT.md to Word/PDF:**

Option 1: Using Pandoc (recommended):
```powershell
# Install pandoc
choco install pandoc

# Convert to PDF
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex -V geometry:margin=1in

# Convert to Word
pandoc REPORT.md -o REPORT.docx
```

Option 2: Manual conversion:
1. Open REPORT.md in VS Code
2. Use Markdown PDF extension to export as PDF
3. Or copy content to Microsoft Word and format

**Add to Report:**

1. **Insert Screenshots:**
   - Add all EDA screenshots in Section 3
   - Add CI/CD screenshots in Section 7
   - Add deployment screenshots in Section 9
   - Add monitoring screenshots in Section 10

2. **Add Links:**
   - Add GitHub repository URL in Section 11
   - Add video demonstration link in Section 11

3. **Update Architecture Diagram:**
   - Replace ASCII diagram with actual image (use draw.io or similar)

4. **Ensure 10 pages:**
   - Add more detail to each section
   - Include code snippets
   - Add more screenshots
   - Expand on challenges and solutions

**Final Report Structure:**
- Cover page with your details
- Table of contents
- All 12 sections from REPORT.md
- Embedded screenshots
- GitHub repository link
- Video demonstration link
- References

---

### Step 10: Final Verification Checklist

Before submission, verify:

- [ ] All model files exist in `models/` folder
- [ ] Dataset exists in `data/` folder
- [ ] API runs successfully locally
- [ ] Docker image builds and runs
- [ ] Kubernetes deployment works
- [ ] Monitoring stack is deployed
- [ ] All screenshots captured
- [ ] GitHub repository is public
- [ ] CI/CD pipeline runs successfully
- [ ] Video demonstration recorded
- [ ] Final report is 10+ pages (Word/PDF)
- [ ] Report includes all screenshots
- [ ] Report includes GitHub link
- [ ] Report includes video link

---

## Quick Reference Commands

```powershell
# Full pipeline
python src/data/download_data.py
python src/data/preprocess.py
python notebooks/01_eda.py
python src/models/train.py
python -m pytest tests/ -v

# API
python -m uvicorn src.api.app:app --reload

# Docker
docker build -t heart-disease-prediction:latest .
docker run -p 8000:8000 heart-disease-prediction:latest

# Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl get pods
kubectl get services

# Monitoring
# Access Grafana at: http://localhost:3000 (admin/admin)
# Access Prometheus at: http://localhost:9090
```

---

## Troubleshooting

**Issue: API fails to load model**
- Solution: Ensure you've run `python src/models/train.py` first

**Issue: Docker build fails**
- Solution: Ensure Docker Desktop is running, check Dockerfile syntax

**Issue: Kubernetes pods not starting**
- Solution: Check `kubectl describe pod <pod-name>` for errors

**Issue: LoadBalancer not working on local**
- Solution: Use NodePort instead, or use Minikube tunnel

**Issue: Grafana can't connect to Prometheus**
- Solution: Check Prometheus service name in Grafana data source configuration

---

## Estimated Time

- Step 1 (Generate files): 15 minutes
- Step 2 (Test API): 10 minutes
- Step 3 (Docker): 15 minutes
- Step 4 (Kubernetes): 30 minutes
- Step 5 (Monitoring): 20 minutes
- Step 6 (Screenshots): 20 minutes
- Step 7 (GitHub): 10 minutes
- Step 8 (Video): 30 minutes
- Step 9 (Report): 1 hour
- **Total: ~3 hours**

---

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Grafana Documentation](https://grafana.com/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
