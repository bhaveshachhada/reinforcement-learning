from typing import Union, Callable

from packages.agents.src.agents.agent import Agent, ObsType, ActType
from packages.environments.src.environments.space import DiscreteSpace
from packages.policies.src.policies.policy import Policy


type Number = Union[int, float]


class QLearningAgent(Agent[ObsType, ActType]):
    def __init__(
        self,
        action_space: DiscreteSpace,
        policy: Policy,
        q_value_getter: Callable[[ObsType, ActType], Number],
        q_value_setter: Callable[[ObsType, ActType, Number], None],
        discount_factor: float,
        learning_rate: float,
    ):
        self.policy = policy
        self.action_space = action_space

        assert isinstance(self.action_space, DiscreteSpace), (
            f"{self.__class__.__name__} only supports DiscreteSpace for Actions"
        )

        self.get_q_value = q_value_getter
        self.set_q_value = q_value_setter
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def choose_action(self, state: ObsType) -> int:
        return self.policy.choose_action(state)

    def _max_q_for_state(self, state: ObsType) -> Number:
        n = self.action_space.n
        start = self.action_space.start
        return max(self.get_q_value(state, a) for a in range(start, start + n))

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
        target = (
            reward
            if done
            else reward + (self.discount_factor * self._max_q_for_state(next_state))
        )
        error = target - current_q
        updated_q = current_q + self.learning_rate * error
        self.set_q_value(state, action, updated_q)
