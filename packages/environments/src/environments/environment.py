from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic

from packages.environments.src.environments.space import Space

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Reward:
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Reward({self.value})"


class Environment(ABC, Generic[ObsType, ActType]):
    action_space: Space[ActType]
    observation_space: Space[ObsType]

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, Reward, bool]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> ObsType:
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError
