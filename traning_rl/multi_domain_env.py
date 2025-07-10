import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.reward_functions import (
    compute_finance_reward,
    compute_education_reward,
    compute_health_reward
)

from data_generators.finance_generator import generate_finance_data
from data_generators.education_generator import generate_education_data
from data_generators.health_generator import generate_health_data

class MultiDomainDataEnv(gym.Env):
    def __init__(self, n_samples=5, openai_api_key=None):
        super().__init__()
        self.n_samples = n_samples
        self.api_key = openai_api_key

        self.domains = ["Finance", "Education", "Health"]
        self.configs = [
            {"realism": "Synthetic", "bias": 0},
            {"realism": "Synthetic", "bias": 20},
            {"realism": "Grounded", "bias": 10},
            {"realism": "Grounded", "bias": 30},
        ]

        self.action_space = spaces.Discrete(len(self.domains) * len(self.configs))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        domain_idx = action // len(self.configs)
        config_idx = action % len(self.configs)

        domain = self.domains[domain_idx]
        config = self.configs[config_idx]

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
        obs = np.array([0.0], dtype=np.float32)
        terminated = True
        truncated = False
        info = {
            "domain": domain,
            "config": config,
            "reward": avg_reward
        }

        return obs, avg_reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

