from types import NoneType
from typing import Union, Callable

import numpy as np

from packages.agents.src.agents.agent import Agent, ObsType, ActType
from packages.environments.src.environments.environment import Environment
from packages.policies.src.policies.policy import Policy


type Number = Union[int, float]


class Sarsa(Agent[ObsType, ActType]):
    def __init__(
        self,
        env: Environment,
        policy: Policy,
        q_value_getter: Callable[[ObsType, ActType], Number],
        q_value_setter: Callable[[ObsType, ActType, Number], NoneType],
        discount_rate: float,
        lr: float,
        rng: np.random.Generator,
    ):
        self.env = env
        self.policy = policy
        self.get_q_value = q_value_getter
        self.set_q_value = q_value_setter
        self.discount_rate = discount_rate
        self.lr = lr
        self.rng = rng

    def choose_action(self, state: ObsType) -> ActType:
        print(f"\033[31m[agent]\033[0m choosing action in {state=}")
        return self.policy.choose_action(state)

    def step(
        self,
        state: ObsType,
        action: ActType,
        reward: Number,
        next_state: ObsType,
        next_action: ActType = None,
        done: bool = False,
    ):
        current_q = self.get_q_value(state, action)
        print(
            f"\033[31m[agent step]\033[0m: {state=}, {action=}, {reward=}, {next_state=}, {next_action=}, {done=}, {current_q=}"
        )
        target = (
            reward
            if done
            else reward
            + (self.discount_rate * self.get_q_value(next_state, next_action))
        )
        error = target - current_q
        updated_q = current_q + (error * self.lr)
        self.set_q_value(state, action, updated_q)
        print(
            f"\033[31m[agent step]\033[0m: {state=}, {action=}, {reward=}, {updated_q=}"
        )
