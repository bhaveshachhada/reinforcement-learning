from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union

from packages.environments.src.environments.environment import Environment
from packages.policies.src.policies.policy import Policy


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class Agent(ABC, Generic[ObsType, ActType]):
    env: Environment
    policy: Policy

    @abstractmethod
    def step(self, action: ActType, reward: Union[int, float]):
        raise NotImplementedError

    def choose_action(self, state: ObsType) -> ActType:
        raise NotImplementedError
