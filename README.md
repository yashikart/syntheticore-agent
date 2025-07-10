# 🧠 SYNTHETICORE-AGENT

**SynthetiCore Agent ** is a smart, feedback-driven synthetic data generation framework designed to create realistic, bias-aware datasets across key domains such as **Finance**, **Education**, and **Health**. It combines **structured simulation**, **LLM-enhanced text generation**, and **reinforcement learning (RL)** to dynamically improve the quality and realism of generated data based on user feedback.

> 🔍 Built to simulate training data for environments like **Uniguru**, **BH Core**, **Financial Sim**, and **Gurukul**.

---

## 🌟 Key Features

- 🏛 **Multi-domain simulation** (Finance, Education, Health)
- 🔁 **RL-based configuration selection** using reward feedback
- 🧠 **GPT-powered summaries**, labels, and natural descriptions
- 🌍 **Language support**: English, Hindi, Hinglish (GPT translation)
- 🧪 Realism toggle: `Synthetic` vs `Grounded` (LLM-enhanced)
- 🧰 Export: CSV, JSON
- 🎙️ Optional: Voice-based prompt input using mic
- 📈 Reward & feedback logging

---

## 📦 Project Structure

```
SYNTHETICORE-AGENTAI/
│
├── app.py                     # Main Streamlit app UI
├── rl_reward_log.csv          # RL reward tracking log
├── feedback_log.csv           # User feedback + sentiment log
│
├── data_generators/           # Domain-wise data generators
│   ├── finance_generator.py
│   ├── education_generator.py
│   ├── health_generator.py
│   └── custom_data_generator.py    # Prompt-based generator
│
├── training_rl/               # PPO-based training modules
│   ├── train_multi_domain.py
│   ├── multi_domain_env.py
│   └── multi_domain_inference.py
│
├── utils/                     # Supporting logic
│   ├── rl_agent.py                 # Rule-based RL agent (non-ppo)
│   ├── reward_functions.py        # Reward logic per domain
│   ├── llm_generator.py           # GPT-based summary generator
│   └── hindi_translator.py        # GPT-based row translation
```

---

## 🖥️ UI Features (`app.py`)

- 🎛 Select dataset domain (Finance / Education / Health)
- 🧠 Toggle realism (Synthetic / GPT-grounded)
- 🌐 Select output language (English / Hindi / Hinglish)
- ⚖️ Adjust bias via slider
- 🔍 Preview data
- 📥 Export as `.csv` or `.json`
- 💬 Submit feedback for each dataset
- 📈 Visualize average reward trend over time

---

## 🤖 Reinforcement Learning Logic

The **RL Agent** uses one of four preset configurations:

```python
configs = [
    {"realism": "Synthetic", "bias": 0},
    {"realism": "Synthetic", "bias": 20},
    {"realism": "Grounded", "bias": 10},
    {"realism": "Grounded", "bias": 30}
]
```

Each time a dataset is generated:

- The agent selects the configuration with the **highest cumulative reward**.
- Reward is computed using:

```python
reward = realism_score + balance_score + variability_score
```

### 🧾 Example – Finance Reward Breakdown

- **Realism**: `Income > Expense`
- **Balance**: `Label ∈ {"Good Saver", "Over-Spender"}`
- **Variability**: Diversity in `Occupation`, `Region`, and `Behavior`

Similar reward logic is applied for **Education** and **Health** domains.  
User feedback is analyzed using **GPT-3.5**, which extracts:

- A **rating** (1–5)
- **Sentiment**: Positive / Negative

These are used to update the RL agent’s reward values dynamically.

---

## 🧪 Domain Modules

Each domain has a dedicated generator function:

### 🏦 Finance

```python
generate_finance_data(realism="Grounded", bias=30)
```

- **Fields**: Income, Expense, Goals, Occupation, Label  
- **Summary**: GPT-generated financial profile

---

### 🎓 Education

```python
generate_education_data(realism="Grounded", bias=20)
```

- **Fields**: Strengths, Weaknesses, Progress  
- **Summary**: GPT-generated feedback & learning strategy

---

### 🏥 Health

```python
generate_health_data(realism="Grounded", bias=10)
```

- **Fields**: Diagnosis, Symptoms, Severity  
- **Summary**: GPT-generated medical case report

> ✅ All domain datasets can be optionally **translated to Hindi or Hinglish** using GPT-powered translation.

---

## ✨ LLM Usage

- 📝 **Summarization**: GPT generates summaries per row or profile
- 🌐 **Translation**: Converts English rows to Hindi/Hinglish
- 📊 **Rating Extraction**: GPT interprets natural feedback to extract numeric scores and sentiment
- 🧾 **Custom Prompt-to-Columns**: Schema inferred from user prompts

---

## 🧠 PPO Agent (Optional)

To enable policy optimization using `stable-baselines3`:

### 🏋️ Train Agent

```bash
cd training_rl
python train_multi_domain.py
```

### 🧠 Run Inference

```bash
python multi_domain_inference.py
```

> Trained model is saved as `ppo_multi_agent.zip`

---

## 📤 Output Examples

- Download formats: `.csv`, `.json`
- Varying columns depending on domain
- Each row includes timestamp, label, summary, and optional translations

---

## 📊 Reward Tracking

- Logged to `rl_reward_log.csv`
- Fields: `timestamp`, `domain`, `file_name`, `avg_reward`, `record_count`
- Visualized using Altair chart in UI

---

## 💬 Feedback Logging

- Logged to `feedback_log.csv`
- Fields: `timestamp`, `domain`, `file_name`, `user_comment`, `GPT_rating`, `sentiment`

---

## 🧑‍🎓 Persona-Driven Simulation

- **📘 Students**: progress, learning style, goal, feedback
- **🏥 Patients**: symptoms, diagnosis, severity, lifestyle
- **💰 Finance Users**: savings goal, spending pattern, occupation

---

## ✅ Deliverables Recap

| Deliverable                          | Status |
| ----------------------------------- | ------ |
| UI with toggles and filters         | ✅     |
| Multi-domain data generators        | ✅     |
| RL reward logic                     | ✅     |
| PPO agent (optional)                | ✅     |
| Feedback + sentiment loop           | ✅     |
| Export (CSV/JSON)                   | ✅     |
| LLM integration (text + translation)| ✅     |
| RL reward visualizations            | ✅     |
| Voice prompt via mic_recorder       | ✅     |

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/syntheticore-agentai.git
cd syntheticore-agentai
pip install -r requirements.txt
```

Set your OpenAI API Key:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

Launch the app:

```bash
streamlit run app.py
```

---

## 📚 Learning Goals Covered

- Prompt engineering
- Data realism & bias simulation
- RL in data pipelines
- GPT integration in structured workflows
- Feedback loop using LLM
- Persona-driven simulation

---

## 👩‍💻 Author

**Yashika Tirkey**  
🎓 Passionate about AI Agents, Data Simulation & Applied LLMs

---

## 🏁 Final Notes

- ✅ Modular architecture for plug-and-play domains
- 🚀 Easy to extend: Add `mental_health_generator.py`, `ecommerce_generator.py`, etc.
- 💬 Optional: Enable Whisper or browser mic for prompt-to-data generation
