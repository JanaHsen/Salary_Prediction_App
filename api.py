from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import os

# Bundle loaded once at startup 
bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.joblib')
    loaded = joblib.load(model_path)
    bundle['model']           = loaded['model']
    bundle['encoders']        = loaded['encoders']
    bundle['experience_map']  = loaded['experience_map']
    bundle['size_map']        = loaded['size_map']
    bundle['job_title_groups']= loaded['job_title_groups']
    bundle['feature_order']   = loaded['feature_order']
    print("Model loaded successfully.")
    yield
    bundle.clear()

app = FastAPI(
    title="Salary Predictor API",
    description="Predicts Data Science salaries using a trained Decision Tree",
    version="1.0.0",
    lifespan=lifespan
)

# ── Helper functions (same logic as Colab) ────────────────────────────────────
def group_job_title(title: str) -> str:
    for group, titles in bundle['job_title_groups'].items():
        if title in titles:
            return group
    return 'Other'

def group_location(loc: str) -> str:
    if loc == 'US':
        return 'US'
    elif loc == 'GB':
        return 'GB'
    return 'Other'

# ── Valid values for input validation ─────────────────────────────────────────
VALID_EXPERIENCE = ['EN', 'MI', 'SE', 'EX']
VALID_COMPANY_SIZE = ['S', 'M', 'L']

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.get("/predict")
def predict(
    experience_level: str = Query(..., description="EN, MI, SE, EX"),
    job_title:        str = Query(..., description="e.g. Data Scientist"),
    company_location: str = Query(..., description="e.g. US, GB, DE"),
    company_size:     str = Query(..., description="S, M, L")
):
    # Validate inputs
    if experience_level not in VALID_EXPERIENCE:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid experience_level '{experience_level}'. Must be one of {VALID_EXPERIENCE}"
        )
    if company_size not in VALID_COMPANY_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid company_size '{company_size}'. Must be one of {VALID_COMPANY_SIZE}"
        )

    # Apply same transforms as training
    job_group      = group_job_title(job_title)
    location_group = group_location(company_location)

    try:
        row = pd.DataFrame([{
            'experience_encoded':        bundle['experience_map'][experience_level],
            'job_title_grouped_encoded': bundle['encoders']['job_title_grouped'].transform([job_group])[0],
            'location_grouped_encoded':  bundle['encoders']['location_grouped'].transform([location_group])[0],
            'size_encoded':              bundle['size_map'][company_size]
        }])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding error: {str(e)}")

    log_pred       = bundle['model'].predict(row)[0]
    predicted_usd  = float(np.expm1(log_pred))

    return {
        "predicted_salary_usd": round(predicted_usd, 2),
        "inputs": {
            "experience_level": experience_level,
            "job_title":        job_title,
            "job_title_group":  job_group,
            "company_location": company_location,
            "location_group":   location_group,
            "company_size":     company_size
        }
    }

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Salary Predictor API is running"}