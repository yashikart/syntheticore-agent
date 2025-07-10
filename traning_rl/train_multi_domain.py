
from stable_baselines3 import PPO
from multi_domain_env import MultiDomainDataEnv
from stable_baselines3.common.env_checker import check_env

env = MultiDomainDataEnv(n_samples=10, openai_api_key="your-api-key")
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1500)
model.save("ppo_multi_agent")
