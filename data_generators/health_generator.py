#health_generator.py

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


def generate_health_data(n=100, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    if realism == "Grounded" and openai_api_key:
        openai.api_key = openai_api_key

    diagnosis_map = {
        'Cold': ['Cough', 'Sneezing', 'Runny Nose', 'Fever'],
        'Flu': ['Fever', 'Fatigue', 'Cough', 'Body Ache', 'Headache'],
        'Migraine': ['Headache', 'Nausea', 'Sensitivity to Light', 'Blurred Vision'],
        'COVID-19': ['Fever', 'Cough', 'Fatigue', 'Loss of Smell', 'Shortness of Breath'],
        'Gastritis': ['Nausea', 'Vomiting', 'Stomach Pain', 'Loss of Appetite'],
        'Asthma': ['Shortness of Breath', 'Wheezing', 'Chest Tightness', 'Cough'],
        'Diabetes': ['Frequent Urination', 'Fatigue', 'Blurred Vision', 'Increased Thirst'],
        'Hypertension': ['Headache', 'Dizziness', 'Nosebleeds', 'Fatigue'],
        'Anemia': ['Fatigue', 'Pale Skin', 'Shortness of Breath', 'Dizziness'],
        'Depression': ['Fatigue', 'Sadness', 'Loss of Interest', 'Sleep Disturbance'],
        'Appendicitis': ['Abdominal Pain', 'Nausea', 'Fever', 'Loss of Appetite'],
        'UTI': ['Burning Sensation', 'Frequent Urination', 'Pelvic Pain', 'Cloudy Urine'],
        'Allergy': ['Sneezing', 'Runny Nose', 'Itchy Eyes', 'Cough'],
        'Chickenpox': ['Fever', 'Rash', 'Itchy Skin', 'Fatigue'],
        'Dengue': ['High Fever', 'Headache', 'Joint Pain', 'Skin Rash']
    }

    for _ in range(n):
        try:
            name = fake.name()
            age = random.randint(1, 90)
            region = fake.city() if bias < 60 else f"Rural {fake.city()}"
            gender = random.choice(["Male", "Female", "Other"])
            diagnosis = random.choice(list(diagnosis_map.keys()))
            symptoms = ", ".join(random.sample(diagnosis_map[diagnosis], 2))
            severity = random.choice(["Mild", "Moderate", "Severe"])
            history = random.choice(["Hypertension", "None", "Asthma"])
            lifestyle = random.choice(["Smoker", "Sedentary", "Athlete"])
            compliance = random.choice(["Adherent", "Irregular"])
            visits = random.randint(1, 5)

            recommended_care = (
                "Home Care" if severity == "Mild"
                else random.choice(["Home Care", "Clinic Visit"]) if severity == "Moderate"
                else "Needs Hospitalization"
            )

            entry = {
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Region": region,
                "Diagnosis": diagnosis,
                "Symptoms": symptoms,
                "Severity": severity,
                "Medical History": history,
                "Lifestyle": lifestyle,
                "Compliance Level": compliance,
                "Visit Count": visits,
                "Recommended Care": recommended_care,
                "Timestamp": pd.Timestamp.now().isoformat()
            }

            if realism == "Grounded" and openai_api_key:
                prompt = (
                    f"Patient {name}, aged {age}, from {region}, has been diagnosed with {diagnosis}. "
                    f"Symptoms: {symptoms}. History: {history}. Lifestyle: {lifestyle}. "
                    f"Compliance: {compliance}. Visits: {visits}. Severity: {severity}. "
                    f"Recommended care: {recommended_care}. Write a brief medical case summary."
                )
                summary = generate_llm_summary(
                    prompt,
                    system_role="You are a medical assistant writing clinical summaries.",
                    temperature=0.5,
                    max_tokens=150
                )
                entry["Case Summary"] = summary
            else:
                entry["Case Summary"] = "Synthetic case summary."

            data.append(entry)

        except Exception as e:
            print(f"Error generating row: {e}")

    df = pd.DataFrame(data)

    if language == "Hindi":
        print("Translating to full Hindi...")
        translated_rows = []
        for _, row in df.iterrows():
            translated_row = {}
            for col, val in row.items():
                val_str = str(val)
                translated_col = translate_to_hindi([col])[0]
                translated_val = translate_to_hindi([val_str])[0] if not val_str.isdigit() else val_str
                translated_row[translated_col] = translated_val
            translated_rows.append(translated_row)
        df = pd.DataFrame(translated_rows)
        print(" Hindi translation complete.")

    elif language == "Hinglish":
        print("🌐 Translating to Hinglish...")
        df_copy = df.copy()
        for col in tqdm(df.columns):
            df_copy[col + "_hinglish"] = df[col].astype(str).apply(hinglish_converter)
        df = df_copy
        print("Hinglish translation complete.")

    if realism == "Synthetic" and "Case Summary" in df.columns:
        df = df.drop(columns=["Case Summary"])

    print(f"Final dataset shape: {df.shape}")

    return df
