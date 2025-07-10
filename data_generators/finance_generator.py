from faker import Faker
import pandas as pd
import random
import openai
from utils.llm_generator import generate_llm_summary


def generate_finance_data(n=100, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    if openai_api_key:
        openai.api_key = openai_api_key

    for _ in range(n):
        name = fake.name()
        region = fake.city() if bias < 60 else f"Rural {fake.city()}"
        income = round(random.uniform(30000, 100000), 2)
        expense = round(random.uniform(income * 0.3, income * 0.9), 2)
        surplus = income - expense
        savings_goal = round(random.uniform(surplus * 0.5, surplus), 2)
        savings_pct = round((surplus / income) * 100, 2)
        occupation = random.choice(["Salaried", "Freelancer", "Business"])
        behavior = random.choice(["Frugal", "Impulsive", "Balanced"])
        goal = random.choice(["Buy a house", "Save for child's education", "Retirement plan"])
        label = "Good Saver" if savings_pct > 20 else "Over-Spender"

        entry = {
            "Name": name,
            "Region": region,
            "Occupation": occupation,
            "Income": income,
            "Expense": expense,
            "Savings Goal": savings_goal,
            "Savings (%)": savings_pct,
            "Spending Behavior": behavior,
            "Financial Goals": goal,
            "Label": label
        }

        if realism == "Grounded" and openai_api_key:
            prompt = (
                f"{name} from {region} is a {occupation} earning ₹{income}, spending ₹{expense}. "
                f"Their goal is to {goal}. Their behavior is described as {behavior}. "
                f"Provide a short financial profile summary."
            )
            entry["Financial Summary"] = generate_llm_summary(prompt, system_role="You are a financial advisor writing a profile summary.")
        else:
            entry["Financial Summary"] = "Synthetic financial profile."

        data.append(entry)

    return pd.DataFrame(data)