from abc import ABC, abstractmethod
from typing import Tuple


class Action:
    pass


class State:
    pass


class Reward:
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Reward({self.value})"


class Environment(ABC):
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, bool]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError
