import requests
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

#  Config 
API_BASE_URL = "http://127.0.0.1:8000"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

#  Sample inputs to cover the input space 
SAMPLE_INPUTS = [
    {"experience_level": "EN", "job_title": "Data Analyst",         "company_location": "US", "company_size": "S"},
    {"experience_level": "MI", "job_title": "Data Scientist",        "company_location": "US", "company_size": "M"},
    {"experience_level": "SE", "job_title": "Data Engineer",         "company_location": "US", "company_size": "L"},
    {"experience_level": "EX", "job_title": "Data Science Manager",  "company_location": "US", "company_size": "L"},
    {"experience_level": "MI", "job_title": "Machine Learning Engineer", "company_location": "GB", "company_size": "M"},
    {"experience_level": "SE", "job_title": "Research Scientist",    "company_location": "GB", "company_size": "L"},
    {"experience_level": "EN", "job_title": "Data Analyst",          "company_location": "DE", "company_size": "S"},
    {"experience_level": "SE", "job_title": "Data Scientist",        "company_location": "DE", "company_size": "M"},
]

#  Step 1: Call FastAPI 
def get_prediction(sample: dict) -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/predict", params=sample, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: FastAPI is not running. Start it with: uvicorn api:app --reload")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: API returned {response.status_code} — {response.text}")
        raise

#  Step 2: Build prompt and call Ollama 
def get_llm_analysis(prediction: dict) -> str:
    salary = prediction['predicted_salary_usd']
    inputs = prediction['inputs']

    experience_labels = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior', 'EX': 'Executive'}
    size_labels = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}

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
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        raise
    except Exception as e:
        print(f"ERROR calling Ollama: {e}")
        raise

#  Step 3: Build result record 
def build_record(prediction: dict, narrative: str) -> dict:
    inputs = prediction['inputs']
    return {
        "predicted_salary_usd": prediction['predicted_salary_usd'],
        "experience_level":     inputs['experience_level'],
        "job_title":            inputs['job_title'],
        "job_title_group":      inputs['job_title_group'],
        "company_location":     inputs['company_location'],
        "location_group":       inputs['location_group'],
        "company_size":         inputs['company_size'],
        "llm_narrative":        narrative,
        "created_at":           datetime.now(timezone.utc).isoformat()
    }

#  Supabase client 
from supabase import create_client

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

def push_to_supabase(record: dict):
    try:
        client = get_supabase_client()
        client.table("predictions").insert(record).execute()
        print(f"  → Pushed to Supabase ✓")
    except Exception as e:
        print(f"  → Supabase push failed: {e}")
        raise

#  Main runner 
def run_pipeline(sample: dict) -> dict:
    print(f"\nRunning: {sample['job_title']} | {sample['experience_level']} | {sample['company_location']}")

    print("  → Calling FastAPI...")
    prediction = get_prediction(sample)
    print(f"  → Predicted salary: ${prediction['predicted_salary_usd']:,.0f}")

    print("  → Calling Ollama...")
    narrative = get_llm_analysis(prediction)
    print(f"  → Narrative received ({len(narrative)} chars)")

    record = build_record(prediction, narrative)

    print("  → Pushing to Supabase...")
    push_to_supabase(record)

    return record

#  Entry point 
if __name__ == "__main__":
    results = []

    for sample in SAMPLE_INPUTS:
        record = run_pipeline(sample)
        results.append(record)

    print(f"\n✓ Pipeline complete. {len(results)} records generated.")
    print("\nSample record:")
    print(json.dumps(results[0], indent=2))