import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openai
import pandas as pd
import random
from faker import Faker
from datetime import datetime

fake = Faker()


def extract_columns_from_prompt(prompt_text, openai_api_key):
    """
    Uses OpenAI LLM to extract column names from user prompt.
    Returns a clean list of column names like: ['Name', 'Age', 'City']
    """
    openai.api_key = openai_api_key
    system_message = (
        "You are a helpful assistant. From the user prompt, extract realistic and relevant column names "
        "for a structured dataset. Only return a comma-separated list like: Name, Age, Income, Region, Goal."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=100,
            temperature=0.3
        )
        cols = response.choices[0].message.content.strip()
        return [col.strip().title() for col in cols.replace(".", "").split(",") if col.strip()]
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return []


def generate_custom_data(columns, n=10):
    """
    Generates custom data based on LLM-inferred column schema.
    Handles basic keyword-based column guessing.
    """
    data = []
    for _ in range(n):
        row = {}
        for col in columns:
            col_lower = col.lower()
            if "name" in col_lower:
                row[col] = fake.name()
            elif "email" in col_lower:
                row[col] = fake.email()
            elif "phone" in col_lower:
                row[col] = fake.phone_number()
            elif "income" in col_lower:
                row[col] = round(random.uniform(30000, 120000), 2)
            elif "expense" in col_lower:
                income = row.get("Income", round(random.uniform(30000, 120000), 2))
                row[col] = round(random.uniform(income * 0.3, income * 0.9), 2)
            elif "age" in col_lower:
                row[col] = random.randint(18, 75)
            elif "gender" in col_lower:
                row[col] = random.choice(["Male", "Female", "Other"])
            elif "goal" in col_lower:
                row[col] = random.choice(["Buy a house", "Start a business", "Retirement plan"])
            elif "region" in col_lower or "city" in col_lower:
                row[col] = fake.city()
            elif "occupation" in col_lower:
                row[col] = random.choice(["Engineer", "Doctor", "Freelancer", "Teacher"])
            elif "date" in col_lower:
                row[col] = fake.date_this_decade()
            elif "timestamp" in col_lower:
                row[col] = datetime.now().isoformat()
            elif "id" in col_lower:
                row[col] = fake.uuid4()
            else:
                row[col] = fake.word().capitalize()
        data.append(row)
    return pd.DataFrame(data)


def compute_reward(df):
    """
    Reward = realism + variability across rows.
    For now, realism checks if Income > Expense if present.
    """
    def score(row):
        realism = 0
        if "Income" in row and "Expense" in row:
            realism += 1 if row["Income"] > row["Expense"] else 0
        variability = len(set(row.values())) / len(row)
        return realism + variability

    return round(df.apply(score, axis=1).mean(), 2)
