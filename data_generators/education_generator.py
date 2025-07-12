# education_generator.py

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
from utils.llm_generator import generate_llm_summary

# Translation helpers
def translate_to_hindi(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

nlp = spacy.load("en_core_web_sm")

def hinglish_converter(text):
    doc = nlp(text)
    key_words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN", "ADV", "NUM"] and token.is_alpha]
    try:
        hindi = translate_to_hindi([text], tokenizer, model)[0]
    except:
        return text  # fallback

    for word in key_words:
        try:
            translated_word = translate_to_hindi([word], tokenizer, model)[0]
            hindi = hindi.replace(translated_word, word)
        except:
            continue

    return hindi


# Main Education Data Generator
def generate_education_data(n=50, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    subjects = ['Math', 'Science', 'English', 'History']
    progress_list = ['Improving', 'Stable', 'Declining']

    if realism == "Grounded" and openai_api_key:
        openai.api_key = openai_api_key

    if language == "Hindi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    else:
        tokenizer = model = None

    for _ in range(n):
        name = fake.name()
        region = fake.city() if bias < 50 else f"Rural {fake.city()}"
        progress = random.choice(progress_list)
        learning_style = random.choice(["Visual", "Auditory", "Kinesthetic"])
        strengths = random.sample(subjects, 2)
        weaknesses = random.choice([s for s in subjects if s not in strengths])
        goal = random.choice(["Pass boards", "Crack JEE", "Improve grades"])

        entry = {
            "Name": name,
            "Region": region,
            "Progress": progress,
            "Learning Style": learning_style,
            "Strengths": ", ".join(strengths),
            "Weak Subjects": weaknesses,
            "Learning Goal": goal
        }

        if realism == "Grounded" and openai_api_key:
            prompt = (
                f"Student {name} from {region} is a {learning_style} learner showing {progress} progress. "
                f"Strengths: {entry['Strengths']}, Weak Subject: {entry['Weak Subjects']}. "
                f"Goal: {goal}. Write a brief academic feedback summary."
            )
            entry["Feedback Summary"] = generate_llm_summary(
                prompt,
                system_role="You are an educational expert providing personalized learning advice.",
                temperature=0.6,
                max_tokens=150,
                api_key=openai_api_key
            )
        else:
            entry["Feedback Summary"] = "Synthetic feedback summary."

        data.append(entry)

    df = pd.DataFrame(data)

    # 🌐 Translate to Hindi
    if language == "Hindi":
        print("🌐 Translating to Hindi...")
        translated_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            translated_row = {}
            for col, val in row.items():
                col_hi = translate_to_hindi([col], tokenizer, model)[0]
                val_hi = translate_to_hindi([str(val)], tokenizer, model)[0] if not isinstance(val, (int, float)) else val
                translated_row[col_hi] = val_hi
            translated_rows.append(translated_row)
        df = pd.DataFrame(translated_rows)
        print("✅ Hindi translation complete.")

    # 🔤 Hinglish conversion
    elif language == "Hinglish":
        print("🔤 Converting to Hinglish...")
        for col in df.columns:
            if df[col].dtype == "object":
                df[col + "_hinglish"] = [hinglish_converter(str(val)) for val in tqdm(df[col])]
        print("✅ Hinglish conversion complete.")

    if realism == "Synthetic" and "Feedback Summary" in df.columns:
        df = df.drop(columns=["Feedback Summary"])

    print("🎓 Education dataset generated with columns:", df.columns.tolist())
    return df
