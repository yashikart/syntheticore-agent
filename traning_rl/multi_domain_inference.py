# traning_rl/multi_domain_inference.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from traning_rl.multi_domain_env import MultiDomainDataEnv
import pandas as pd

# Step 1: Initialize environment with OpenAI API key
env = MultiDomainDataEnv(openai_api_key="your-openai-api-key")  # <-- REPLACE with your key

# Step 2: Load trained PPO model
model = PPO.load("ppo_multi_agent")

# Step 3: Reset environment and predict
obs, _ = env.reset(seed=42)
action, _ = model.predict(obs, deterministic=True)

# Step 4: Decode recommended domain + config
domain_idx = action // 4
config_idx = action % 4
domain = env.domains[domain_idx]
config = env.configs[config_idx]

print(f"PPO Recommendation → Domain: {domain}, Config: {config}")

#  Step 5: Generate data using recommended domain + config
if domain == "Education":
    from data_generators.education_generator import generate_education_data
    df = generate_education_data(n=10, realism=config["realism"], bias=config["bias"], openai_api_key="your-openai-api-key")
elif domain == "Finance":
    from data_generators.finance_generator import generate_finance_data
    df = generate_finance_data(n=10, realism=config["realism"], bias=config["bias"], openai_api_key="your-openai-api-key")
elif domain == "Health":
    from data_generators.health_generator import generate_health_data
    df = generate_health_data(n=10, realism=config["realism"], bias=config["bias"], openai_api_key="your-openai-api-key")
else:
    raise ValueError("Invalid domain received from PPO.")

# Step 6: Save and preview
output_file = f"ppo_generated_{domain.lower()}.csv"
df.to_csv(output_file, index=False)
print(f"📁 Data saved to {output_file}")
print(df.head())
