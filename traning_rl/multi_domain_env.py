# rl_env_multi.py
import gym
import numpy as np
from gym import spaces
from utils.reward_functions import compute_finance_reward, compute_education_reward, compute_health_reward

from data_generators.finance_generator import generate_finance_data
from data_generators.education_generator import generate_education_data
from data_generators.health_generator import generate_health_data

class MultiDomainDataEnv(gym.Env):
    def __init__(self, n_samples=5, openai_api_key=None):
        super(MultiDomainDataEnv, self).__init__()
        self.n_samples = n_samples
        self.api_key = openai_api_key

        self.domains = ["Finance", "Education", "Health"]
        self.configs = [
            {"realism": "Synthetic", "bias": 0},
            {"realism": "Synthetic", "bias": 20},
            {"realism": "Grounded", "bias": 10},
            {"realism": "Grounded", "bias": 30},
        ]

        # Action space: domain index (0–2) × config index (0–3) = 12 discrete actions
        self.action_space = spaces.Discrete(len(self.domains) * len(self.configs))

        # Observation space (unused)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        domain_idx = action // len(self.configs)
        config_idx = action % len(self.configs)

        domain = self.domains[domain_idx]
        config = self.configs[config_idx]

        # Generate data
        if domain == "Finance":
            df = generate_finance_data(n=self.n_samples, realism=config["realism"], bias=config["bias"], openai_api_key=self.api_key)
            rewards = df.apply(compute_finance_reward, axis=1)
        elif domain == "Education":
            df = generate_education_data(n=self.n_samples, realism=config["realism"], bias=config["bias"], openai_api_key=self.api_key)
            rewards = df.apply(compute_education_reward, axis=1)
        else:
            df = generate_health_data(n=self.n_samples, realism=config["realism"], bias=config["bias"], openai_api_key=self.api_key)
            rewards = df.apply(compute_health_reward, axis=1)

        avg_reward = float(np.mean(rewards))
        done = True
        next_state = np.array([0.0], dtype=np.float32)

        info = {
            "domain": domain,
            "config": config,
            "reward": avg_reward
        }

        return next_state, avg_reward, done, info
