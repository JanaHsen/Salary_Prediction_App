import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

@st.cache_data(ttl=30)
def load_predictions():
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        client = create_client(url, key)
        response = client.table("predictions").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load predictions from Supabase: {e}")
        return pd.DataFrame()
    
#  Config 
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Data Science Salary Dashboard",
    page_icon="💼",
    layout="wide"
)

#  Load raw data for EDA visuals 
@st.cache_data
def load_data():
    df = pd.read_csv('ds_salaries.csv')
    df = df.drop(columns=['Unnamed: 0', 'salary', 'salary_currency'])
    df = df.drop_duplicates()

    experience_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
    size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    df['experience_label'] = df['experience_level'].map(experience_map)
    df['size_label'] = df['company_size'].map(size_map)

    def group_location(loc):
        if loc == 'US': return 'US'
        elif loc == 'GB': return 'GB'
        return 'Other'

    df['location_group'] = df['company_location'].apply(group_location)
    return df

df = load_data()

#  Sidebar  prediction inputs 
st.sidebar.title("Predict Your Salary")
st.sidebar.markdown("Fill in your details below:")

experience = st.sidebar.selectbox(
    "Experience Level",
    options=['EN', 'MI', 'SE', 'EX'],
    format_func=lambda x: {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}[x]
)

job_title = st.sidebar.selectbox(
    "Job Title",
    options=[
        'Data Scientist', 'Data Engineer', 'Data Analyst',
        'Machine Learning Engineer', 'Research Scientist',
        'Data Science Manager', 'ML Engineer', 'Data Architect',
        'Applied Data Scientist', 'Principal Data Scientist'
    ]
)

location = st.sidebar.selectbox(
    "Company Location",
    options=['US', 'GB', 'DE', 'IN', 'CA', 'FR', 'ES', 'Other'],
)

company_size = st.sidebar.selectbox(
    "Company Size",
    options=['S', 'M', 'L'],
    format_func=lambda x: {'S': 'Small', 'M': 'Medium', 'L': 'Large'}[x]
)

predict_btn = st.sidebar.button("Predict Salary", type="primary")

#  Main layout 
st.title("💼 Data Science Salary Landscape")
st.markdown("*Exploring who earns what, why, and what the data reveals.*")
st.divider()

#  Section 1: Dataset overview 
st.subheader("The Landscape")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",    f"{len(df):,}")
col2.metric("Median Salary",    f"${df['salary_in_usd'].median():,.0f}")
col3.metric("Highest Salary",   f"${df['salary_in_usd'].max():,.0f}")
col4.metric("Unique Job Titles", df['job_title'].nunique())

st.divider()

#  Section 2: Salary distribution 
st.subheader("Salary Distribution")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df['salary_in_usd'], bins=40, color='steelblue', edgecolor='white')
    ax.set_title('Raw Salary Distribution')
    ax.set_xlabel('Salary (USD)')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(df['salary_in_usd']), bins=40, color='coral', edgecolor='white')
    ax.set_title('Log-Transformed Distribution')
    ax.set_xlabel('log(1 + Salary)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.close()

st.divider()

#  Section 3: Key drivers 
st.subheader("What Drives Salary?")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ['Entry', 'Mid', 'Senior', 'Executive']
    sns.boxplot(data=df, x='experience_label', y='salary_in_usd',
                order=order, palette='Blues', ax=ax)
    ax.set_title('Salary by Experience Level')
    ax.set_xlabel('Experience')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x='location_group', y='salary_in_usd',
                order=['US', 'GB', 'Other'], palette='Greens', ax=ax)
    ax.set_title('Salary by Location')
    ax.set_xlabel('Location Group')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()

st.divider()

#  Section 4: The surprise 
st.subheader("The Surprise")
st.markdown("*Does company size actually matter as much as we assume?*")

fig, ax = plt.subplots(figsize=(12, 4))
pivot = df.groupby(['size_label', 'experience_label'])['salary_in_usd'].median().unstack()
pivot = pivot.reindex(columns=['Entry', 'Mid', 'Senior', 'Executive'])
pivot = pivot.reindex(['Small', 'Medium', 'Large'])
pivot.plot(kind='bar', ax=ax, colormap='Blues', edgecolor='white')
ax.set_title('Median Salary by Company Size and Experience Level')
ax.set_xlabel('Company Size')
ax.set_ylabel('Median Salary (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.legend(title='Experience', bbox_to_anchor=(1.01, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
st.pyplot(fig)
plt.close()

st.divider()

# ─ Section 5: Prediction result 
st.subheader("Your Prediction")

if predict_btn:
    with st.spinner("Calling API and generating analysis..."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/predict",
                params={
                    "experience_level": experience,
                    "job_title": job_title,
                    "company_location": location,
                    "company_size": company_size
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            salary = result['predicted_salary_usd']

            # Show predicted salary
            st.success(f"### Predicted Salary: ${salary:,.0f} USD")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Input Summary**")
                st.json(result['inputs'])

            with col2:
                # Show prediction in context of similar profiles
                similar = df[
                    (df['experience_level'] == experience) &
                    (df['location_group'] == ('US' if location == 'US' else 'GB' if location == 'GB' else 'Other'))
                ]['salary_in_usd']

                if len(similar) > 0:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(similar, bins=20, color='steelblue',
                            edgecolor='white', alpha=0.7, label='Similar profiles')
                    ax.axvline(salary, color='red', linewidth=2,
                               linestyle='--', label=f'Your prediction: ${salary:,.0f}')
                    ax.set_title('Your Prediction vs Similar Profiles')
                    ax.set_xlabel('Salary (USD)')
                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it with: uvicorn api:app --reload")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("Fill in your details in the sidebar and click **Predict Salary** to see results.")

# ── Section 6: Prediction history from Supabase ───────────────────────────────
st.subheader("Prediction History")

predictions_df = load_predictions()

if predictions_df.empty:
    st.info("No predictions yet. Use the sidebar to generate your first prediction.")
else:
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(predictions_df))
    col2.metric("Avg Predicted Salary",
                f"${predictions_df['predicted_salary_usd'].mean():,.0f}")
    col3.metric("Highest Prediction",
                f"${predictions_df['predicted_salary_usd'].max():,.0f}")

    # Latest prediction narrative
    st.markdown("**Latest Analysis**")
    latest = predictions_df.iloc[0]
    st.markdown(f"*{latest['job_title']} | {latest['experience_level']} | {latest['company_location']}*")
    st.markdown(latest['llm_narrative'])

    # History table
    st.markdown("**All Predictions**")
    display_cols = ['created_at', 'job_title', 'experience_level',
                    'company_location', 'company_size', 'predicted_salary_usd']
    st.dataframe(
        predictions_df[display_cols].rename(columns={
            'created_at': 'Date',
            'job_title': 'Job Title',
            'experience_level': 'Experience',
            'company_location': 'Location',
            'company_size': 'Size',
            'predicted_salary_usd': 'Predicted Salary'
        }),
        use_container_width=True
    )

#  Footer 
st.divider()
st.caption(f"Data: ds_salaries.csv — {len(df)} records | Model: Decision Tree Regressor | R²: 0.41")