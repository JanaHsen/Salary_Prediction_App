from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import requests
import os

# ── Bundle loaded once at startup ─────────────────────────────────────────────
bundle = {}

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.joblib')
    loaded = joblib.load(model_path)
    bundle['model']            = loaded['model']
    bundle['encoders']         = loaded['encoders']
    bundle['experience_map']   = loaded['experience_map']
    bundle['size_map']         = loaded['size_map']
    bundle['job_title_groups'] = loaded['job_title_groups']
    bundle['feature_order']    = loaded['feature_order']
    print("Model loaded successfully.")
    yield
    bundle.clear()

app = FastAPI(
    title="Salary Predictor API",
    description="Predicts Data Science salaries using a trained Decision Tree",
    version="1.0.0",
    lifespan=lifespan
)

# ── Helper functions ──────────────────────────────────────────────────────────
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

def get_llm_narrative(inputs: dict, salary: float) -> str:
    experience_labels = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior', 'EX': 'Executive'}
    size_labels       = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}

    prompt = f"""You are a data science compensation analyst. Analyze this salary prediction:

Role: {inputs['job_title']} (category: {inputs['job_title_group']})
Experience: {experience_labels[inputs['experience_level']]}
Location: {inputs['company_location']} (market: {inputs['location_group']})
Company Size: {size_labels[inputs['company_size']]}
Predicted Salary: ${salary:,.0f} USD

Write exactly 3 paragraphs:
1. Whether ${salary:,.0f} is competitive for a {experience_labels[inputs['experience_level']]} {inputs['job_title']} in {inputs['company_location']}
2. Which specific factors are pushing this salary up or down
3. One concrete insight or surprising finding about this profile

Be specific. Use the exact salary figure. Do not confuse experience level with location or company size."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"LLM analysis unavailable: {str(e)}"

# ── Valid values ──────────────────────────────────────────────────────────────
VALID_EXPERIENCE   = ['EN', 'MI', 'SE', 'EX']
VALID_COMPANY_SIZE = ['S', 'M', 'L']

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.get("/predict")
def predict(
    experience_level: str = Query(..., description="EN, MI, SE, EX"),
    job_title:        str = Query(..., description="e.g. Data Scientist"),
    company_location: str = Query(..., description="e.g. US, GB, DE"),
    company_size:     str = Query(..., description="S, M, L")
):
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

    log_pred      = bundle['model'].predict(row)[0]
    predicted_usd = float(np.expm1(log_pred))

    inputs = {
        "experience_level": experience_level,
        "job_title":        job_title,
        "job_title_group":  job_group,
        "company_location": company_location,
        "location_group":   location_group,
        "company_size":     company_size
    }

    # Call Ollama for narrative
    narrative = get_llm_narrative(inputs, predicted_usd)

    return {
        "predicted_salary_usd": round(predicted_usd, 2),
        "llm_narrative":        narrative,
        "inputs":               inputs
    }

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Salary Predictor API is running"}