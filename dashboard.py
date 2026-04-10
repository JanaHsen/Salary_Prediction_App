import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Data Science Salary Dashboard",
    page_icon="💼",
    layout="wide"
)

# ── Load raw data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv('ds_salaries.csv')

@st.cache_data
def load_clean_data():
    df = pd.read_csv('ds_salaries.csv')
    df = df.drop(columns=['Unnamed: 0', 'salary', 'salary_currency'])
    df = df.drop_duplicates()
    experience_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
    size_map       = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    df['experience_label'] = df['experience_level'].map(experience_map)
    df['size_label']       = df['company_size'].map(size_map)
    def group_location(loc):
        if loc == 'US': return 'US'
        elif loc == 'GB': return 'GB'
        return 'Other'
    df['location_group'] = df['company_location'].apply(group_location)
    return df

@st.cache_data(ttl=30)
def load_predictions():
    try:
        url    = os.getenv("SUPABASE_URL")
        key    = os.getenv("SUPABASE_KEY")
        client = create_client(url, key)
        response = client.table("predictions").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load predictions: {e}")
        return pd.DataFrame()

raw_df   = load_raw_data()
clean_df = load_clean_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💼 Predict Your Salary")
st.sidebar.markdown("Fill in your profile:")

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
    options=['US', 'GB', 'DE', 'IN', 'CA', 'FR', 'ES', 'Other']
)
company_size = st.sidebar.selectbox(
    "Company Size",
    options=['S', 'M', 'L'],
    format_func=lambda x: {'S': 'Small', 'M': 'Medium', 'L': 'Large'}[x]
)
predict_btn = st.sidebar.button("Predict Salary", type="primary")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("💼 Data Science Salary Landscape")
st.markdown("*Exploring who earns what, why, and what the data reveals.*")
st.divider()

# ── Section 1: Raw Dataset ────────────────────────────────────────────────────
st.subheader("The Raw Data")
st.markdown("This is what the dataset looked like before any cleaning:")
st.dataframe(raw_df.head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Columns kept:**")
    st.markdown("""
- `experience_level` — seniority of the role
- `job_title` — specific role name
- `salary_in_usd` — our target variable
- `company_location` — where the company is based
- `company_size` — S / M / L
- `remote_ratio` — kept for EDA, dropped from model
- `work_year` — kept for EDA, dropped from model
""")
with col2:
    st.markdown("**Columns removed and why:**")
    st.markdown("""
- `Unnamed: 0` — just a row index, no value
- `salary` — raw salary in local currency, not comparable
- `salary_currency` — redundant after converting to USD
- `employee_residence` — highly correlated with company location
- `employment_type` — 97% full-time, no variance
""")

st.markdown(f"After removing duplicates: **{len(clean_df)} unique records** from {len(raw_df)} original rows.")
st.divider()

# ── Section 2: EDA ────────────────────────────────────────────────────────────
st.subheader("What the Data Tells Us")

# Row 1: Experience and Location
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ['Entry', 'Mid', 'Senior', 'Executive']
    sns.boxplot(data=clean_df, x='experience_label', y='salary_in_usd',
                order=order, palette='Blues', ax=ax)
    ax.set_title('Salary by Experience Level')
    ax.set_xlabel('Experience')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()
    st.caption("Clear progression — experience is a strong salary driver.")

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=clean_df, x='location_group', y='salary_in_usd',
                order=['US', 'GB', 'Other'], palette='Greens', ax=ax)
    ax.set_title('Salary by Location')
    ax.set_xlabel('Location Group')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()
    st.caption("US median $135k vs GB $78k vs Other $59k — location dominates.")

# Row 2: Company size and Year
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=clean_df, x='size_label', y='salary_in_usd',
                order=['Small', 'Medium', 'Large'], palette='Oranges', ax=ax)
    ax.set_title('Salary by Company Size')
    ax.set_xlabel('Company Size')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    st.pyplot(fig)
    plt.close()
    st.caption("Some impact — but weaker than location or experience.")

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    clean_df.groupby('work_year')['salary_in_usd'].median().plot(
        kind='bar', color='steelblue', edgecolor='white', ax=ax)
    ax.set_title('Median Salary by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Median Salary (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)
    plt.close()
    st.caption("Upward trend exists — but only 3 years of data. Dropped from model.")

st.divider()

# ── Section 3: Prediction ─────────────────────────────────────────────────────
st.subheader("Your Salary Prediction")

if predict_btn:
    with st.spinner("Calling model and generating analysis... this may take up to 2 minutes."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/predict",
                params={
                    "experience_level": experience,
                    "job_title":        job_title,
                    "company_location": location,
                    "company_size":     company_size
                },
                timeout=200
            )
            response.raise_for_status()
            result    = response.json()
            salary    = result['predicted_salary_usd']
            narrative = result['llm_narrative']

            # Salary display
            st.success(f"### Predicted Salary: ${salary:,.0f} USD")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Your Profile**")
                exp_labels  = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
                size_labels = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
                st.markdown(f"""
- **Role:** {job_title}
- **Experience:** {exp_labels[experience]}
- **Location:** {location}
- **Company Size:** {size_labels[company_size]}
""")

            with col2:
                # Prediction in context
                loc_group = 'US' if location == 'US' else 'GB' if location == 'GB' else 'Other'
                similar   = clean_df[
                    (clean_df['experience_level'] == experience) &
                    (clean_df['location_group'] == loc_group)
                ]['salary_in_usd']

                if len(similar) > 0:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(similar, bins=20, color='steelblue',
                            edgecolor='white', alpha=0.7, label='Similar profiles')
                    ax.axvline(salary, color='red', linewidth=2,
                               linestyle='--', label=f'Your prediction')
                    ax.set_title('Where You Land')
                    ax.set_xlabel('Salary (USD)')
                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()

            # LLM Narrative
            st.markdown("---")
            st.markdown("**AI Market Analysis**")
            st.markdown(narrative)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it with: uvicorn api:app --reload")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("Fill in your profile in the sidebar and click **Predict Salary**.")

st.divider()

# ── Section 4: Prediction History ────────────────────────────────────────────
st.subheader("Prediction History")

predictions_df = load_predictions()

if predictions_df.empty:
    st.info("No predictions yet.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(predictions_df))
    col2.metric("Avg Predicted Salary",
                f"${predictions_df['predicted_salary_usd'].mean():,.0f}")
    col3.metric("Highest Prediction",
                f"${predictions_df['predicted_salary_usd'].max():,.0f}")

    display_cols = ['created_at', 'job_title', 'experience_level',
                    'company_location', 'company_size', 'predicted_salary_usd']
    st.dataframe(
        predictions_df[display_cols].rename(columns={
            'created_at':            'Date',
            'job_title':             'Job Title',
            'experience_level':      'Experience',
            'company_location':      'Location',
            'company_size':          'Size',
            'predicted_salary_usd':  'Predicted Salary'
        }),
        use_container_width=True
    )

st.divider()
st.caption(f"Data: ds_salaries.csv — {len(clean_df)} records | Model: Decision Tree Regressor | R²: 0.41")