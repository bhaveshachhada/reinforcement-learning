import numpy as np
import pytest

from packages.agents.src.agents.sarsa import Sarsa


class StubPolicy:
    def __init__(self, action):
        self.action = action

    def choose_action(self, state):
        return self.action


def make_agent(q_table, discount_rate=0.9, lr=0.1, policy=None):
    return Sarsa(
        policy=policy or StubPolicy(action=0),
        q_value_getter=lambda state, action: q_table.get((state, action), 0.0),
        q_value_setter=lambda state, action, value: q_table.__setitem__(
            (state, action), value
        ),
        discount_rate=discount_rate,
        lr=lr,
        rng=np.random.default_rng(0),
    )


class TestChooseAction:
    def test_delegates_to_policy(self):
        agent = make_agent(q_table={}, policy=StubPolicy(action=3))
        assert agent.choose_action(state="s0") == 3


class TestStep:
    def test_updates_q_value_with_discounted_next_q_when_not_done(self):
        q_table = {("s0", "a0"): 2.0, ("s1", "a1"): 5.0}
        agent = make_agent(q_table, discount_rate=0.9, lr=0.1)

        agent.step(
            state="s0",
            action="a0",
            reward=1.0,
            next_state="s1",
            next_action="a1",
            done=False,
        )

        assert q_table[("s0", "a0")] == pytest.approx(2.35)

    def test_updates_q_value_using_only_reward_when_done(self):
        q_table = {("s0", "a0"): 3.0}
        agent = make_agent(q_table, discount_rate=0.9, lr=0.5)

        agent.step(
            state="s0",
            action="a0",
            reward=10.0,
            next_state=None,
            next_action=None,
            done=True,
        )

        assert q_table[("s0", "a0")] == pytest.approx(6.5)

    def test_unvisited_state_action_defaults_to_zero_q_value(self):
        q_table = {}
        agent = make_agent(q_table, discount_rate=0.5, lr=1.0)

        agent.step(
            state="s0",
            action="a0",
            reward=4.0,
            next_state="s1",
            next_action="a1",
            done=False,
        )

        assert q_table[("s0", "a0")] == pytest.approx(4.0)
