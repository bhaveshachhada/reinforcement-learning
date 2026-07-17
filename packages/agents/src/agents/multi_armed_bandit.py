from types import NoneType
from typing import Union

import numpy as np

from packages.agents.src.agents.agent import Agent
from packages.environments.src.environments.multi_armed_bandit import (
    MultiArmedBanditEnvironment,
)
from packages.policies.src.policies.policy import Policy


class MultiArmedBanditAgent(Agent[None, int]):
    def __init__(
        self,
        env: MultiArmedBanditEnvironment,
        policy: Policy,
        q_values: np.ndarray[tuple[int], np.float64],
        rng: np.random.Generator,
    ):
        self.env = env
        self.policy = policy

        self.n_arms = env.n_arms
        self.q_values = q_values
        self.arm_pull_count: np.ndarray[tuple[int], np.uint8] = np.zeros(
            self.n_arms, dtype=np.uint64
        )
        self.rng = rng

    def choose_action(self, state: None) -> int:
        return self.policy.choose_action(state)

    def step(
        self,
        state: NoneType,
        action: int,
        reward: Union[int, float],
        next_state: NoneType,
        next_action: NoneType = None,
        done: bool = False,
    ):
        old_q_value = self.q_values[action]
        step_size = 1 / max(1, self.arm_pull_count[action])
        new_q_value = old_q_value + (step_size * (reward - old_q_value))
        self.arm_pull_count[action] += 1
        self.q_values[action] = new_q_value
