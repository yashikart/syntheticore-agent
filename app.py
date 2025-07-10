import os
import streamlit as st
import pandas as pd
import re
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
from faker import Faker
from googletrans import Translator
import openai

from data_generators.finance_generator import generate_finance_data
from data_generators.education_generator import generate_education_data
from data_generators.health_generator import generate_health_data
from data_generators.custom_data_generator import extract_columns_from_prompt, generate_fake_data, compute_reward
from utils.reward_functions import log_reward, visualize_reward_trends, log_feedback_entry
from utils.rl_agent import RLDataAgent
from utils.hindi_translator import translate_dataframe_to_hindi

fake = Faker()

st.set_page_config(page_title="SynthetiCore Agent", layout="wide", page_icon="🧠")
st.title("🧠 SynthetiCore Agent")
st.caption("🔬 LLM-RL Synthetic Dataset Generator")

# ------------------ Session State ------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "👋 Hello! Ask me to generate datasets like `Generate 30 grounded finance records in Hindi`"}]
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# ------------------ Sidebar Settings ------------------
with st.sidebar:
    st.header("🔐 OpenAI Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get your key](https://platform.openai.com/account/api-keys)")

    st.header("🛠 Customize Dataset")
    dataset_type = st.selectbox("📂 Domain", ["Finance", "Health", "Education", "Custom"])
    realism = st.radio("🧬 Realism", ["Synthetic", "Grounded"], horizontal=True)
    language = st.selectbox("🌐 Language", ["English", "Hindi", "Hinglish"])
    bias = st.slider("🌯 Bias (%)", 0, 100, 0)
    num_records = st.slider("🔹 Number of Records", 5, 100, 10)

    if dataset_type == "Custom":
        prompt = st.text_area("🖍️ Describe the custom dataset (e.g., income, expense, goal, occupation, etc.)")

    if st.button("🚀 Generate Dataset"):
        with st.spinner("Generating data..."):
            if dataset_type == "Finance":
                agent = RLDataAgent(domain="Finance")
                rl_config = agent.get_next_config()
                df = generate_finance_data(num_records, rl_config["realism"], language, rl_config["bias"], openai_api_key)
            elif dataset_type == "Health":
                df = generate_health_data(num_records, realism, language, bias, openai_api_key)
            elif dataset_type == "Education":
                df = generate_education_data(num_records, realism, language, bias, openai_api_key)
            elif dataset_type == "Custom":
                extracted_columns = extract_columns_from_prompt(prompt, openai_api_key)
                if not extracted_columns:
                    st.warning("No valid columns found from your description.")
                    st.stop()
                df = generate_fake_data(extracted_columns, num_records)
                reward = compute_reward(df)
                st.info(f"📈 Reward Score: `{reward}`")

            if language.lower() == "hindi":
                df = translate_dataframe_to_hindi(df)

            file_name = f"{dataset_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(file_name, index=False)
            log_reward(dataset_type, df, file_name)

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"📦 Your **{dataset_type}** dataset with **{num_records}** records is ready!",
                "file": file_name
            })

# ------------------ Chat Display ------------------
for i, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "file" in msg:
            with open(msg["file"], "rb") as f:
                st.download_button("⬇️ Download CSV", f, file_name=msg["file"], mime="text/csv", key=f"dl_{i}")
            st.markdown("📊 **Please rate the usefulness of this dataset:**")
            rating = st.radio("Rating (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5], horizontal=True, key=f"rating_{i}")
            feedback_comment = st.text_area("📜 Additional Comments", key=f"feedback_comment_{i}")
            if st.button("✅ Submit Feedback", key=f"submit_{i}"):
                rating_value = log_feedback_entry(feedback_comment, dataset_type, msg.get("file", "N/A"), openai_api_key)
                
                if rating_value is not None and dataset_type == "Finance":
                    agent = RLDataAgent(domain="Finance")
                    agent.update_policy_from_feedback(rating_value)
                
                st.success("🎉 Feedback submitted! Thank you.")


# ------------------ Chat Input ------------------
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.chat_input("e.g., Generate 50 synthetic health records in Hindi")
with col2:
    audio = mic_recorder(start_prompt="🎧", stop_prompt="✅", key="audio_input")
final_input = user_input or (audio["text"] if audio else None)

# ------------------ Handle Commands ------------------
if final_input:
    if not openai_api_key:
        st.info("Please enter your OpenAI API key.")
        st.stop()

    openai.api_key = openai_api_key
    st.session_state.chat_messages.append({"role": "user", "content": final_input})

    domain = "Finance"
    lang = "English"
    realism_type = "Synthetic"
    count = 10

    if "hindi" in final_input.lower():
        lang = "Hindi"
    elif "hinglish" in final_input.lower():
        lang = "Hinglish"

    if "health" in final_input.lower():
        domain = "Health"
    elif "education" in final_input.lower():
        domain = "Education"
    elif "custom" in final_input.lower():
        domain = "Custom"

    if "grounded" in final_input.lower():
        realism_type = "Grounded"

    count_match = re.search(r"\d+", final_input)
    if count_match:
        count = int(count_match.group())

    with st.spinner("Building dataset..."):
        if domain == "Finance":
            agent = RLDataAgent(domain="Finance")
            rl_config = agent.get_next_config()
            df = generate_finance_data(count, rl_config["realism"], lang, rl_config["bias"], openai_api_key)
        elif domain == "Health":
            df = generate_health_data(count, realism_type, lang, bias, openai_api_key)
        elif domain == "Education":
            df = generate_education_data(count, realism_type, lang, bias, openai_api_key)
        else:
            extracted_columns = extract_columns_from_prompt(final_input, openai_api_key)
            df = generate_fake_data(extracted_columns, count)

        if lang.lower() == "hindi":
            df = translate_dataframe_to_hindi(df)

        file_name = f"{domain.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)
        log_reward(domain, df, file_name)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"✅ Here's your **{realism_type} {domain}** dataset in **{lang}** with **{count} records**.",
            "file": file_name
        })

# ------------------ Reward Visualization ------------------
with st.expander("📈 View Reward Trends"):
    visualize_reward_trends(st)
