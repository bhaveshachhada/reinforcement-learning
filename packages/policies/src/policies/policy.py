from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic

import numpy as np

from packages.environments.src.environments.space import Space, DiscreteSpace

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Policy(ABC, Generic[ObsType, ActType]):
    @abstractmethod
    def choose_action(self, state: ObsType) -> ActType:
        raise NotImplementedError


class ConstantActionPolicy(Policy[ObsType, ActType]):
    def __init__(self, action: ActType):
        self.action = action

    def choose_action(self, state: ObsType) -> ActType:
        return self.action


class RandomActionPolicy(Policy[ObsType, ActType]):
    def __init__(self, action_space: Space[ActType]):
        self.action_space = action_space

    def choose_action(self, state: ObsType) -> ActType:
        return self.action_space.sample()


class GreedyPolicy(Policy[ObsType, int]):
    def __init__(
        self,
        action_space: DiscreteSpace,
        q_value_fn: Callable[[ObsType, int], float],
        rng: np.random.Generator,
    ):
        self.action_space = action_space
        self.q_value_fn = q_value_fn
        self.rng = rng

    def choose_action(self, state: ObsType) -> int:
        actions = range(
            self.action_space.start, self.action_space.start + self.action_space.n
        )
        q_values = [self.q_value_fn(state, action) for action in actions]
        best_value = max(q_values)
        best_actions = [
            action for action, value in zip(actions, q_values) if value == best_value
        ]
        return self.rng.choice(best_actions)


class EpsilonGreedyPolicy(Policy[ObsType, int]):
    def __init__(
        self,
        action_space: DiscreteSpace,
        q_value_fn: Callable[[ObsType, int], float],
        epsilon: float,
        rng: np.random.Generator,
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be between 0 and 1, but got {}".format(epsilon)
            )
        self.action_space = action_space
        self.epsilon = epsilon
        self.rng = rng
        self.greedy_policy = GreedyPolicy(action_space, q_value_fn, rng)

    def choose_action(self, state: ObsType) -> int:
        if self.rng.random() < self.epsilon:
            return self.action_space.sample()
        return self.greedy_policy.choose_action(state)
