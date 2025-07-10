import altair as alt
import pandas as pd
import openai
import os

def compute_finance_reward(entry):
    realism = 1 if entry["Income"] > entry["Expense"] else 0
    balance = 1 if entry["Label"] in ["Good Saver", "Over-Spender"] else 0
    variability = len(set([entry["Occupation"], entry["Spending Behavior"], entry["Region"]])) / 3
    return realism + balance + variability

def compute_education_reward(entry):
    realism = 1 if 0 <= int(entry["Math"]) <= 100 and 0 <= int(entry["English"]) <= 100 else 0
    balance = 1 if entry["Progress"] in ["Improving", "Stable", "Declining"] else 0
    variability = len(set([entry["Learning Style"], entry["Strengths"], entry["Weak Subjects"]])) / 3
    return realism + balance + variability

def compute_health_reward(entry):
    realism = 1 if entry["Severity"] in ["Mild", "Moderate", "Severe"] and isinstance(entry["Symptoms"], str) else 0
    balance = 1 if entry["Recommended Care"] in ["Home Care", "Needs Hospitalization"] else 0
    variability = len(set([entry["Medical History"], entry["Lifestyle"], entry["Compliance Level"]])) / 3
    return realism + balance + variability

def log_reward(domain, df, file_name):
    if domain == "Finance":
        rewards = df.apply(compute_finance_reward, axis=1)
    elif domain == "Education":
        rewards = df.apply(compute_education_reward, axis=1)
    elif domain == "Health":
        rewards = df.apply(compute_health_reward, axis=1)
    else:
        rewards = [0] * len(df)

    avg_reward = round(pd.Series(rewards).mean(), 3)

    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "domain": domain,
        "file": file_name,
        "records": len(df),
        "avg_reward": avg_reward
    }

    reward_log_file = "rl_reward_log.csv"
    if os.path.exists(reward_log_file):
        existing = pd.read_csv(reward_log_file)
        new_df = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        new_df = pd.DataFrame([log_entry])
    new_df.to_csv(reward_log_file, index=False)

def visualize_reward_trends(st):
    reward_log_file = "rl_reward_log.csv"

    if os.path.exists(reward_log_file):
        reward_df = pd.read_csv(reward_log_file)
        reward_df["timestamp"] = pd.to_datetime(reward_df["timestamp"])

        st.subheader("📈 Average Reward Trend by Domain")
        color_scale = alt.Scale(
            domain=["Finance", "Health", "Education"],
            range=["#1f77b4", "#2ca02c", "#ff7f0e"]
        )

        line_chart = alt.Chart(reward_df).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('avg_reward:Q', title='Average Reward', scale=alt.Scale(domain=[0, 3])),
            color=alt.Color('domain:N', scale=color_scale, title='Domain'),
            tooltip=['timestamp:T', 'domain:N', 'avg_reward:Q']
        ).properties(
            width='container',
            height=320
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("No reward log found yet. Please generate a dataset first.")

def extract_feedback_rating(comment, api_key):
    openai.api_key = api_key
    prompt = f"Rate this user comment on a scale from 1 (worst) to 5 (excellent):\n\n\"{comment}\"\n\nOnly respond with a single number."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a feedback rating assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0.0
        )
        rating = int(response.choices[0].message.content.strip())
        return min(max(rating, 1), 5)
    except:
        return None

def extract_feedback_sentiment(comment, api_key):
    openai.api_key = api_key
    prompt = f"Classify the sentiment of this comment as Positive or Negative:\n\n\"{comment}\"\n\nRespond with only one word: Positive or Negative."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment classifier."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0.0
        )
        sentiment = response.choices[0].message.content.strip().capitalize()
        if sentiment in ["Positive", "Negative"]:
            return sentiment
        return "Neutral"
    except:
        return "Neutral"

def log_feedback_entry(comment, domain, file_name, api_key):
    rating = extract_feedback_rating(comment, api_key)
    sentiment = extract_feedback_sentiment(comment, api_key)

    if not rating:
        return None

    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "domain": domain,
        "file": file_name,
        "feedback": comment,
        "rating": rating,
        "sentiment": sentiment
    }

    feedback_file = "feedback_log.csv"
    if os.path.exists(feedback_file):
        old = pd.read_csv(feedback_file)
        new = pd.concat([old, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        new = pd.DataFrame([log_entry])
    new.to_csv(feedback_file, index=False)

    return rating
