import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
from faker import Faker
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

fake = Faker()

def parse_generation_prompt(prompt_text, openai_api_key):
    """
    Parses prompts like 'Generate 5 grounded finance records in Hinglish'
    or 'Create dataset with name, age, income...'

    Returns:
        - domain: 'Finance', 'Health', 'Education', or 'Custom'
        - columns: list of str (for custom)
        - realism: 'Synthetic' or 'Grounded'
        - language: 'English', 'Hindi', or 'Hinglish'
        - rows: int
    """
    system_msg = (
        "You're a dataset schema extractor. From user prompts, extract:\n"
        "- domain (finance, health, education, custom)\n"
        "- columns (only if custom)\n"
        "- realism (synthetic or grounded)\n"
        "- language (english, hindi, hinglish)\n"
        "- row count (e.g., 5)\n"
        "Return in valid JSON: {\"domain\":..., \"columns\":..., \"realism\":..., \"language\":..., \"rows\":...}"
    )

    try:
        chat = ChatOpenAI(api_key=openai_api_key, temperature=0.2, model="gpt-3.5-turbo")
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt_text)
        ]
        response = chat.invoke(messages)
        return eval(response.content)
    except Exception as e:
        print(f"❌ Prompt parsing error: {e}")
        return None
