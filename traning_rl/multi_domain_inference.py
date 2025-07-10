
from stable_baselines3 import PPO
from multi_domain_env import MultiDomainDataEnv

env = MultiDomainDataEnv(openai_api_key="your-api-key")
model = PPO.load("ppo_multi_agent")

obs = env.reset()
action, _ = model.predict(obs, deterministic=True)

domain_idx = action // 4
config_idx = action % 4
domain = env.domains[domain_idx]
config = env.configs[config_idx]

print(f"✅ Best Recommendation from PPO → Domain: {domain}, Config: {config}")
