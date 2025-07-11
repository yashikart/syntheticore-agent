import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
from faker import Faker
from transformers import MarianTokenizer, MarianMTModel
import openai
import spacy
from tqdm import tqdm
from googletrans import Translator
from utils.llm_generator import generate_llm_summary

# Hindi translation helper
def translate_to_hindi(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

# Hinglish transliteration helper
nlp = spacy.load("en_core_web_sm")
translator = Translator()

def hinglish_converter(text):
    doc = nlp(text)
    key_words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN", "ADV", "NUM"] and token.is_alpha]
    hindi = translator.translate(text, src='en', dest='hi').text
    for word in key_words:
        try:
            translated_word = translator.translate(word, src='en', dest='hi').text
            hindi = hindi.replace(translated_word, word)
        except:
            continue
    return hindi

# Main Finance Data Generator
def generate_finance_data(n=100, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    # Load model if Hindi
    if language == "Hindi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    else:
        tokenizer = model = None

    if realism == "Grounded" and openai_api_key:
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
                f"{name} from {region} is a {occupation} earning ₹{income} and spending ₹{expense}. "
                f"Their goal is to {goal}. Their behavior is {behavior}. "
                f"Write a brief financial profile summary."
            )
            summary = generate_llm_summary(
                prompt,
                system_role="You are a financial assistant generating profile summaries.",
                temperature=0.5,
                max_tokens=120,
                api_key=openai_api_key
            )
            entry["Financial Summary"] = summary
        else:
            entry["Financial Summary"] = "Synthetic financial profile."

        data.append(entry)

    df = pd.DataFrame(data)

    # Translate dataset
    if language == "Hindi":
        print("Translating to Hindi...")
        translated_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            translated_row = {}
            for col, val in row.items():
                col_hi = translate_to_hindi([col], tokenizer, model)[0]
                val_hi = translate_to_hindi([str(val)], tokenizer, model)[0] if not isinstance(val, (int, float)) else val
                translated_row[col_hi] = val_hi
            translated_rows.append(translated_row)
        df = pd.DataFrame(translated_rows)
        print("Hindi translation complete.")

    elif language == "Hinglish":
        print("Converting to Hinglish...")
        for col in df.columns:
            if df[col].dtype == "object":
                df[col + "_hinglish"] = [hinglish_converter(str(x)) for x in tqdm(df[col])]
        print("Hinglish columns added.")

    print(f"Final dataset shape: {df.shape}")
    return df
