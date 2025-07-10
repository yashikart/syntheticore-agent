from faker import Faker
import random
import pandas as pd
import openai
from utils.llm_generator import generate_llm_summary


def translate_row_with_llm(row, target_language="Hindi"):
    """
    Translates a row of data to Hindi or Hinglish using LLM.
    Only string fields are translated.
    """
    translation_prompt = (
        f"Translate this patient data row to {target_language}. "
        f"Keep numbers and format intact. Return JSON format only.\n\n"
        f"{row.to_dict()}"
    )

    translated_json = generate_llm_summary(
        translation_prompt,
        system_role="You are a medical data translator. Only return the translated JSON.",
        temperature=0.3,
        max_tokens=300
    )

    try:
        translated_dict = eval(translated_json)  # Safe here only because model outputs valid JSON
        return translated_dict
    except Exception as e:
        print(f"⚠️ Error parsing translation: {e}")
        return row.to_dict()


def generate_health_data(n=100, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

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

    if openai_api_key:
        openai.api_key = openai_api_key

    for i in range(n):
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

            if severity == "Mild":
                recommended_care = "Home Care"
            elif severity == "Moderate":
                recommended_care = random.choice(["Home Care", "Clinic Visit"])
            else:
                recommended_care = "Needs Hospitalization"

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
                english_prompt = (
                    f"Patient {name}, aged {age}, from {region}, has been diagnosed with {diagnosis}. "
                    f"Symptoms include: {symptoms}. Medical history: {history}. Lifestyle: {lifestyle}. "
                    f"Compliance: {compliance}. Number of visits: {visits}. Severity is {severity}, "
                    f"so care recommendation is {recommended_care}. Write a brief medical case summary."
                )
                summary = generate_llm_summary(
                    english_prompt,
                    system_role="You are a multilingual medical assistant generating case summaries.",
                    temperature=0.7,
                    max_tokens=150
                )
                entry["Case Summary"] = summary
            else:
                entry["Case Summary"] = "Synthetic case summary."

            data.append(entry)

        except Exception as e:
            print(f"⚠️ Error in entry {i}: {e}")

    df = pd.DataFrame(data)

    # 🔁 Translate entire dataset to Hindi or Hinglish
    if language in ["Hindi", "Hinglish"] and openai_api_key:
        translated_rows = []
        for idx, row in df.iterrows():
            translated = translate_row_with_llm(row, target_language=language)
            translated_rows.append(translated)
        df = pd.DataFrame(translated_rows)

    print("✅ Generated Columns:", df.columns.tolist())
    return df
