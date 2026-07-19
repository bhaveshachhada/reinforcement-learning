import numpy as np
import pytest

from packages.agents.src.agents.multi_armed_bandit import MultiArmedBanditAgent
from packages.environments.src.environments.multi_armed_bandit import (
    MultiArmedBanditEnvironment,
)


class StubPolicy:
    def __init__(self, action):
        self.action = action

    def choose_action(self, state):
        return self.action


def make_agent(n_arms=3, policy=None):
    env = MultiArmedBanditEnvironment(n_arms=n_arms, rng=np.random.default_rng(0))
    q_values = np.zeros((n_arms,), dtype=np.float64)
    return MultiArmedBanditAgent(
        n_arms=env.n_arms,
        policy=policy or StubPolicy(action=0),
        q_values=q_values,
        rng=np.random.default_rng(0),
    )


class TestChooseAction:
    def test_delegates_to_policy(self):
        agent = make_agent(policy=StubPolicy(action=2))
        assert agent.choose_action(state=None) == 2


class TestStep:
    def test_first_pull_sets_q_value_to_the_observed_reward(self):
        agent = make_agent(n_arms=3)
        agent.step(state=None, action=1, reward=5.0, next_state=None)
        assert agent.q_values[1] == pytest.approx(5.0)
        assert agent.arm_pull_count[1] == 1

    def test_second_pull_on_same_arm_replaces_q_value_with_latest_reward(self):
        agent = make_agent(n_arms=3)
        agent.step(state=None, action=1, reward=5.0, next_state=None)
        agent.step(state=None, action=1, reward=9.0, next_state=None)
        assert agent.q_values[1] == pytest.approx(9.0)
        assert agent.arm_pull_count[1] == 2

    def test_third_pull_averages_with_previous_q_value(self):
        agent = make_agent(n_arms=3)
        agent.step(state=None, action=1, reward=10.0, next_state=None)
        agent.step(state=None, action=1, reward=20.0, next_state=None)
        agent.step(state=None, action=1, reward=30.0, next_state=None)
        assert agent.q_values[1] == pytest.approx(25.0)
        assert agent.arm_pull_count[1] == 3

    def test_pulling_one_arm_does_not_affect_other_arms(self):
        agent = make_agent(n_arms=3)
        agent.step(state=None, action=0, reward=100.0, next_state=None)
        assert agent.q_values[1] == pytest.approx(0.0)
        assert agent.q_values[2] == pytest.approx(0.0)
        assert agent.arm_pull_count[1] == 0
        assert agent.arm_pull_count[2] == 0
