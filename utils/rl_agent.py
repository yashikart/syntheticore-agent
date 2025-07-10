# ------------------ rl_agent.py ------------------
import os
import json

class RLDataAgent:
    def __init__(self, domain="Finance"):
        self.domain = domain
        self.reward_log_path = f"logs/rewards_{domain.lower()}.json"

        self.configs = [
            {"realism": "Synthetic", "bias": 0},
            {"realism": "Synthetic", "bias": 20},
            {"realism": "Grounded", "bias": 10},
            {"realism": "Grounded", "bias": 30},
        ]

        self.rewards = [0] * len(self.configs)
        self.last_index = 0

        self._load_rewards()

    def get_next_config(self):
        max_reward = max(self.rewards)
        best_indices = [i for i, r in enumerate(self.rewards) if r == max_reward]
        self.last_index = best_indices[0] if best_indices else 0
        return self.configs[self.last_index]

    def update_policy_from_feedback(self, rating):
        if rating is None:
            return
        self.rewards[self.last_index] += rating
        self._save_rewards()

    def _load_rewards(self):
        if os.path.exists(self.reward_log_path):
            try:
                with open(self.reward_log_path, "r") as f:
                    self.rewards = json.load(f)
            except Exception as e:
                print(f"[RLAgent] Error loading rewards: {e}")
                self.rewards = [0] * len(self.configs)

    def _save_rewards(self):
        try:
            os.makedirs(os.path.dirname(self.reward_log_path), exist_ok=True)
            with open(self.reward_log_path, "w") as f:
                json.dump(self.rewards, f)
        except Exception as e:
            print(f"[RLAgent] Error saving rewards: {e}")
