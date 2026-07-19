from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic, Union

from packages.policies.src.policies.policy import Policy


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class Agent(ABC, Generic[ObsType, ActType]):
    policy: Policy

    @abstractmethod
    def step(
        self,
        state: ObsType,
        action: ActType,
        reward: Union[int, float],
        next_state: ObsType,
        next_action: Optional[ActType] = None,
        done: bool = False,
    ):
        raise NotImplementedError

    def choose_action(self, state: ObsType) -> ActType:
        raise NotImplementedError
