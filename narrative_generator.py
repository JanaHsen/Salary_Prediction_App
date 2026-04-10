from supabase import create_client
from dotenv import load_dotenv
import requests
import os

load_dotenv()

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def generate_narrative(row: dict) -> str:
    experience_labels = {
        'EN': 'Entry-level', 'MI': 'Mid-level',
        'SE': 'Senior', 'EX': 'Executive'
    }
    size_labels = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    salary = row['predicted_salary_usd']

    prompt = f"""You are a data science compensation analyst. Analyze this salary prediction:

Role: {row['job_title']} (category: {row['job_title_group']})
Experience: {experience_labels[row['experience_level']]}
Location: {row['company_location']} (market: {row['location_group']})
Company Size: {size_labels[row['company_size']]}
Predicted Salary: ${salary:,.0f} USD

Write exactly 3 paragraphs:
1. Whether ${salary:,.0f} is competitive for this profile
2. Which factors are pushing this salary up or down
3. One concrete insight or surprising finding

Be specific. Use the exact salary figure."""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180
    )
    return response.json()['response']

def run():
    client = get_supabase_client()

    rows = client.table("predictions")\
        .select("*")\
        .is_("llm_narrative", "null")\
        .execute()

    if not rows.data:
        print("No rows without narrative found.")
        return

    print(f"Found {len(rows.data)} rows without narrative.")

    for row in rows.data:
        print(f"  → {row['job_title']} | {row['experience_level']} | {row['company_location']}")
        narrative = generate_narrative(row)
        client.table("predictions")\
            .update({"llm_narrative": narrative})\
            .eq("id", row['id'])\
            .execute()
        print(f"  → Saved ✓")

    print(f"\n✓ Done.")

if __name__ == "__main__":
    run()