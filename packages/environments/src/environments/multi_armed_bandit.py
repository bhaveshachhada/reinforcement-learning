from typing import Tuple

import numpy as np

from packages.environments.src.environments.environment import Environment, RewardType
from packages.environments.src.environments.space import DiscreteSpace


class MultiArmedBanditEnvironment(Environment[None, int]):
    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms: int = n_arms
        self.rng = rng
        self.action_space = DiscreteSpace(self.n_arms, 0, self.rng)
        self.observation_space = None

        # random reward mean values with mean 0, variance 1
        self.reward_mean_values = self.rng.normal(loc=20, scale=5, size=(n_arms,))

    def step(self, action: int) -> Tuple[None, RewardType, bool]:
        reward = self.reward_mean_values[action] + self.rng.normal(loc=0, scale=3)
        return None, reward, False

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
