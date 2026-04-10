# Data Science Salary Prediction App

## Overview
An end-to-end ML pipeline that predicts Data Science salaries using a 
trained Decision Tree Regressor, served through a FastAPI endpoint, 
analyzed by a local LLM, and displayed on a Streamlit dashboard.

## Architecture
Local Pipeline → Supabase → Streamlit Dashboard

## Model Performance
- Algorithm: Decision Tree Regressor
- Test R²: 0.41
- Test MAE: ~$36,000
- Training rows: 452
- Features: experience level, job title group, location group, company size

## Why R² is 0.41
The dataset contains 565 unique records across 50+ job titles and 
50+ countries. A single Decision Tree on this data volume has a hard 
performance ceiling. The 0.41 score was achieved through:
- Principled feature engineering backed by EDA
- Log transformation of the skewed target variable
- Grid search hyperparameter tuning
- Dropping zero-importance features (remote ratio, work year)

## Tech Stack
- Model: scikit-learn DecisionTreeRegressor
- API: FastAPI + Uvicorn
- LLM: Ollama (llama3.2:3b)
- Storage: Supabase
- Dashboard: Streamlit

## Setup
1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Add `.env` with Supabase credentials
5. Start API: `uvicorn api:app --reload`
6. Run pipeline: `python pipeline.py`
7. Launch dashboard: `streamlit run dashboard.py`

## Dataset
Kaggle: Data Science Job Salaries
565 unique records after deduplication