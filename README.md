
# 🧠 SyntheticCore Agent UI

SyntheticCore Agent is an intelligent dataset generation system that allows users to create **customized synthetic or grounded datasets** across multiple domains like **Finance, Education, and Health**. It uses **Faker for synthetic data**, **LLMs for grounded text**, **Transformer models for Hindi & Hinglish translation**, and trains a **Reinforcement Learning (RL) agent using PPO** to improve data generation automatically over time.

---

## ⚙️ Requirements

Make sure you have the following installed:

```bash
pip install -r requirements.txt
```

Environment file (`.env`):

```bash
OPENAI_API_KEY=your_openai_key_here
```

Python version: `>=3.8`

Used libraries:

- `streamlit`
- `transformers`
- `gymnasium`
- `stable-baselines3`
- `faker`
- `pandas`, `numpy`, `altair`

---

## 🧰 How Dataset Generation Works

### 🛠️ 1. Synthetic Dataset (Faker)

We use the **Faker** library to generate realistic synthetic data. Depending on the selected domain, the app creates mock values such as:

- **Finance** → income, expense, occupation, region, etc.  
- **Education** → progress, subjects, learning style, etc.  
- **Health** → symptoms, severity, medical history, etc.

Faker ensures that the generated data is **structured and controllable**, but not necessarily based on real-world facts.

---

### 🌍 2. Grounded Dataset (LLMs)

When the user selects **Grounded realism**, we use **OpenAI’s LLM** to generate real-world-like context. For example:

- Health: generates realistic **case summaries**  
- Finance: generates **financial behavior summaries**  
- Education: generates **feedback summaries**

These grounded texts are generated via GPT (using `openai.ChatCompletion`) and are closer to human-written descriptions.

---

### 🌐 3. Language Support (English, Hindi, Hinglish)

We support multilingual dataset generation using **Transformer models**:

#### ✅ Hindi:
- Used **`MarianMTModel`** and **`MarianTokenizer`** from Hugging Face
- Model: `Helsinki-NLP/opus-mt-en-hi`
- Example: "Headache, Fever" → "सिरदर्द, बुखार"

#### ✅ Hinglish:
- Used a **custom mixture** of English + Hindi Transformer output
- Strategy: Keep structure in English and insert **nouns/adjectives in Hindi**
- Makes it friendly and relatable for casual use

---

## 🏗️ Customize Your Dataset

The user can customize the dataset by selecting:

- **Domain**: Finance, Health, Education  
- **Realism**: Synthetic or Grounded  
- **Language**: English, Hindi, Hinglish  
- **Bias Level**: 0–100%  
- **Number of Rows**: 1 to N  

---

## 📈 Reward Function Logic

After generating the dataset, we evaluate it using a **domain-specific reward function**:

```python
Reward = Realism + Balance + Variability
```

- **Realism** → Are the values meaningful and contextually correct?  
- **Balance** → Does the data contain a good mix (e.g., gender, region, etc.)?  
- **Variability** → How diverse is the dataset?

The total average reward is shown in the UI with 💡, and logged into `rl_reward_log.csv`.

---

## 💬 User Feedback Loop

After each dataset is generated:

- Bot asks user to rate (1–5) and leave feedback  
- Comment + Rating is saved in `feedback_log.csv`  
- Sentiment is optionally analyzed using LLM

---

## 🤖 Reinforcement Learning Logic

To **automatically improve** the dataset generation:

### 🧪 Custom Gym Environment

We created `MultiDomainDataEnv` with:

- Action Space → All combinations of (Domain, Realism, Bias)
- Observation → Dummy value
- Reward → Computed based on the dataset quality

### 🔁 PPO Training

We used `Proximal Policy Optimization (PPO)` from Stable Baselines3 to train the RL agent.

Training steps:

1. Agent selects an action → e.g., “Education, Grounded, Bias=30%”
2. A dataset is generated using that config
3. Reward is computed
4. PPO updates its policy to favor better configs

Training script: `train_multi_domain.py`  
Saved model: `ppo_multi_agent.zip`

---

### ⚡ Inference Using Trained Agent

Once trained:

- Load the model via `multi_domain_inference.py`
- The agent will recommend the best config
- Dataset will be generated automatically
- User feedback continues to be logged to improve the agent

---

## 📋 Chatbot Prompt Example

```plaintext
User: What is the difference between synthetic and grounded datasets?

Bot: Synthetic datasets use Faker to generate mock data. Grounded datasets are created using real-world-like summaries powered by LLMs.

User: I want 3 rows with health data, grounded, in Hindi, bias 50.

Bot: [Generated dataset]
Bot: Please rate this dataset (1-5) and leave a comment.
```

---

## 📊 Logs & Visualization

- `rl_reward_log.csv` → Tracks rewards over time  
- `feedback_log.csv` → Stores comments + ratings  
- 📈 Altair charts used for reward trends per domain in the UI

---

## 📎 Summary

SyntheticCore Agent intelligently generates data, collects feedback, and learns from it using reinforcement learning. It's ideal for simulating domain-specific datasets, analyzing user preferences, and improving generation quality with every interaction.
