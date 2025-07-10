from faker import Faker
import pandas as pd
import random
import openai
from utils.llm_generator import generate_llm_summary

def generate_education_data(n=50, realism="Synthetic", language="English", bias=0, openai_api_key=None):
    fake = Faker()
    data = []

    subjects = ['Math', 'Science', 'English', 'History']
    progress_list = ['Improving', 'Stable', 'Declining']

    if openai_api_key:
        openai.api_key = openai_api_key

    for _ in range(n):
        name = fake.name()
        progress = random.choice(progress_list)
        learning_style = random.choice(["Visual", "Auditory", "Kinesthetic"])
        strengths = random.sample(subjects, 2)
        weaknesses = random.choice([s for s in subjects if s not in strengths])
        goal = random.choice(["Pass boards", "Crack JEE", "Improve grades"])

        entry = {
            "Name": name,
            "Progress": progress,
            "Learning Style": learning_style,
            "Strengths": ", ".join(strengths),
            "Weak Subjects": weaknesses,
            "Learning Goal": goal
        }

        if realism == "Grounded" and openai_api_key:
            prompt = (
                f"Student {name} is a {learning_style} learner making {progress} progress. "
                f"Strengths: {entry['Strengths']}, Weaknesses: {entry['Weak Subjects']}. "
                f"Goal: {goal}. Provide feedback and a learning strategy summary."
            )
            entry["Feedback Summary"] = generate_llm_summary(prompt, system_role="You are an educational consultant generating feedback.")
        else:
            entry["Feedback Summary"] = "Synthetic feedback summary."

        data.append(entry)

    return pd.DataFrame(data)
