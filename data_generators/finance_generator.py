# finance_generator.py

import pandas as pd
import random
from faker import Faker
import openai
import spacy
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from utils.llm_generator import generate_llm_summary

# Load SpaCy for POS tagging
nlp = spacy.load("en_core_web_sm")

# Load MarianMT for English → Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_to_hindi(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]


def hinglish_converter(text):
    doc = nlp(text)
    important_tags = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"]
    keywords = [token.text for token in doc if token.pos_ in important_tags and token.is_alpha]

    try:
        hindi_text = translate_to_hindi([text])[0]
        for word in keywords:
            translated_word = translate_to_hindi([word])[0]
            hindi_text = hindi_text.replace(translated_word, word)
        return hindi_text
    except Exception as e:
        print(f"Hinglish conversion error: {e}")
        return text


def generate_finance_data(n=100, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    if realism == "Grounded" and openai_api_key:
        openai.api_key = openai_api_key

    for _ in range(n):
        try:
            name = fake.name()
            region = fake.city() if bias < 60 else f"Rural {fake.city()}"
            income = round(random.uniform(30000, 100000), 2)
            expense = round(random.uniform(income * 0.3, income * 0.9), 2)
            surplus = income - expense
            savings_goal = round(random.uniform(surplus * 0.5, surplus), 2)
            savings_pct = round((surplus / income) * 100, 2)
            occupation = random.choice(["Salaried", "Freelancer", "Business"])
            behavior = random.choice(["Frugal", "Impulsive", "Balanced"])
            goal = random.choice(["Buy a house", "Save for child’s education", "Retirement plan"])
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
                    f"{name} from {region} is a {occupation} earning ₹{income} and spending ₹{expense}. "
                    f"Their goal is to {goal}. Their behavior is {behavior}. "
                    f"Write a brief financial profile summary."
                )
                summary = generate_llm_summary(
                    prompt,
                    system_role="You are a financial assistant generating profile summaries.",
                    temperature=0.5,
                    max_tokens=120
                )
                entry["Financial Summary"] = summary
            else:
                entry["Financial Summary"] = "Synthetic financial profile."

            data.append(entry)

        except Exception as e:
            print(f"Error generating row: {e}")

    df = pd.DataFrame(data)

    if language == "Hindi":
        print("🔁 Translating to Hindi...")
        translated_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            translated_row = {}
            for col, val in row.items():
                col_hi = translate_to_hindi([col])[0]
                val_hi = translate_to_hindi([str(val)])[0] if not isinstance(val, (int, float)) else val
                translated_row[col_hi] = val_hi
            translated_rows.append(translated_row)
        df = pd.DataFrame(translated_rows)
        print("✅ Hindi translation complete.")

    elif language == "Hinglish":
        print("🌐 Translating to Hinglish...")
        df_copy = df.copy()
        for col in tqdm(df.columns):
            df_copy[col + "_hinglish"] = df[col].astype(str).apply(hinglish_converter)
        df = df_copy
        print("✅ Hinglish translation complete.")

    if realism == "Synthetic" and "Financial Summary" in df.columns:
        df = df.drop(columns=["Financial Summary"])

    print(f"✅ Final dataset shape: {df.shape}")
    return df
