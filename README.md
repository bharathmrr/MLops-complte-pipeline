# MLops-complte-pipeline
this project cover enire mlops pipepine and using dvc and aws s3
# ⚙️ MLOps End-to-End Pipeline

This project demonstrates an **end-to-end MLOps workflow** integrating version control, reproducible training, CI/CD, model tracking, and deployment.

Built using:
- ✅ MLFlow for experiment tracking
- ✅ DVC for data versioning
- ✅ FastAPI for model serving
- ✅ Docker for containerization
- ✅ GitHub Actions for CI/CD
- ✅ Monitoring with Prometheus + Grafana

---

## 🧩 Project Architecture

```mermaid
graph TD
  A[Raw Data] -->|DVC| B[Preprocessing]
  B --> C[Feature Engineering]
  C --> D[Train Model]
  D --> E[MLflow Tracking]
  E --> F[Model Registry]
  F --> G[FastAPI Server]
  G --> H[Docker + CI/CD]
  H --> I[Production API]
  I --> J[Monitoring with Prometheus/Grafana]

