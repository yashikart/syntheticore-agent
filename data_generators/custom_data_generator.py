# -------------------- custom_data_generator.py --------------------
import openai
import pandas as pd
import random
from faker import Faker
from datetime import datetime

fake = Faker()

def extract_columns_from_prompt(prompt_text, openai_api_key):
    openai.api_key = openai_api_key
    system_message = (
        "You are a helpful assistant. Given a user's description of a dataset, extract only the most relevant and realistic column names. "
        "Return a comma-separated list like: Name, Age, Income, Expense, Region, Financial Goal."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=100,
            temperature=0.2
        )
        cols = response.choices[0].message.content.strip()
        return [col.strip().title() for col in cols.replace(".", "").split(",") if col.strip()]
    except Exception as e:
        return []

def generate_fake_data(columns, n=5):
    data = []
    for _ in range(n):
        row = {}
        for col in columns:
            col_lower = col.lower()
            if "name" in col_lower:
                row[col] = fake.name()
            elif "income" in col_lower:
                row[col] = round(random.uniform(30000, 100000), 2)
            elif "expense" in col_lower:
                income = row.get("Income", round(random.uniform(30000, 100000), 2))
                row[col] = round(random.uniform(income * 0.3, income * 0.9), 2)
            elif "goal" in col_lower:
                row[col] = random.choice(["Buy a house", "Retirement", "Start a business"])
            elif "region" in col_lower or "city" in col_lower:
                row[col] = fake.city()
            elif "occupation" in col_lower:
                row[col] = random.choice(["Engineer", "Teacher", "Manager", "Student", "Freelancer"])
            elif "age" in col_lower:
                row[col] = random.randint(18, 70)
            elif "gender" in col_lower:
                row[col] = random.choice(["Male", "Female", "Other"])
            elif "date" in col_lower:
                row[col] = fake.date_this_year()
            elif "id" in col_lower:
                row[col] = fake.uuid4()
            else:
                row[col] = fake.word().capitalize()
        data.append(row)
    return pd.DataFrame(data)

def compute_reward(df):
    def score(row):
        realism = 1
        if "Income" in row and "Expense" in row:
            realism += 1 if row["Income"] > row["Expense"] else 0
        variability = len(set(row.values())) / len(row)
        return realism + variability
    return round(df.apply(score, axis=1).mean(), 2)