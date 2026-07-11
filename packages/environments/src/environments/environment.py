from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Union

from packages.environments.src.environments.space import Space

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
type RewardType = Union[int, float]


class Environment(ABC, Generic[ObsType, ActType]):
    action_space: Space[ActType]
    observation_space: Space[ObsType]

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, RewardType, bool]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> ObsType:
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError
