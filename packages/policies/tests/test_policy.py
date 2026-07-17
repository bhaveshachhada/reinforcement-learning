import numpy as np
import pytest

from packages.environments.src.environments.space import DiscreteSpace
from packages.policies.src.policies.policy import (
    ConstantActionPolicy,
    EpsilonGreedyPolicy,
    GreedyPolicy,
    RandomActionPolicy,
)


class StubSpace:
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class TestConstantActionPolicy:
    def test_choose_action_always_returns_configured_action(self):
        policy = ConstantActionPolicy(action=3)
        assert policy.choose_action(state=None) == 3
        assert policy.choose_action(state="anything") == 3


class TestRandomActionPolicy:
    def test_choose_action_delegates_to_action_space_sample(self):
        policy = RandomActionPolicy(action_space=StubSpace(value=7))
        assert policy.choose_action(state=None) == 7

    def test_choose_action_stays_within_action_space(self):
        space = DiscreteSpace(n=4, start=0, rng=np.random.default_rng(0))
        policy = RandomActionPolicy(action_space=space)
        for _ in range(50):
            assert space.contains(policy.choose_action(state=None))


class TestGreedyPolicy:
    def test_choose_action_returns_the_single_best_action(self):
        action_space = DiscreteSpace(n=4, start=0, rng=np.random.default_rng(0))
        q_values = {0: 1.0, 1: 5.0, 2: 2.0, 3: 0.5}
        policy = GreedyPolicy(
            action_space=action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=np.random.default_rng(0),
        )
        assert policy.choose_action(state=None) == 1

    def test_choose_action_breaks_ties_among_best_actions(self):
        action_space = DiscreteSpace(n=5, start=0, rng=np.random.default_rng(0))
        q_values = {0: 0.0, 1: 3.0, 2: 3.0, 3: 1.0, 4: 3.0}
        rng = np.random.default_rng(123)
        policy = GreedyPolicy(
            action_space=action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
        )

        expected = np.random.default_rng(123).choice([1, 2, 4])
        assert policy.choose_action(state=None) == expected

    def test_choose_action_respects_action_space_start_offset(self):
        action_space = DiscreteSpace(n=3, start=10, rng=np.random.default_rng(0))
        q_values = {10: 0.0, 11: 9.0, 12: 1.0}
        policy = GreedyPolicy(
            action_space=action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=np.random.default_rng(0),
        )
        assert policy.choose_action(state=None) == 11


class TestEpsilonGreedyPolicy:
    def test_raises_value_error_for_epsilon_below_zero(self):
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(
                action_space=DiscreteSpace(n=2, start=0, rng=np.random.default_rng(0)),
                q_value_fn=lambda state, action: 0.0,
                epsilon=-0.1,
                rng=np.random.default_rng(0),
            )

    def test_raises_value_error_for_epsilon_above_one(self):
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(
                action_space=DiscreteSpace(n=2, start=0, rng=np.random.default_rng(0)),
                q_value_fn=lambda state, action: 0.0,
                epsilon=1.1,
                rng=np.random.default_rng(0),
            )

    def test_epsilon_zero_always_chooses_greedy_action(self):
        action_space = DiscreteSpace(n=3, start=0, rng=np.random.default_rng(0))
        q_values = {0: 0.0, 1: 9.0, 2: 1.0}
        policy = EpsilonGreedyPolicy(
            action_space=action_space,
            q_value_fn=lambda state, action: q_values[action],
            epsilon=0.0,
            rng=np.random.default_rng(0),
        )
        for _ in range(20):
            assert policy.choose_action(state=None) == 1

    def test_epsilon_one_always_chooses_randomly(self):
        action_space = DiscreteSpace(n=3, start=0, rng=np.random.default_rng(0))
        q_values = {0: 0.0, 1: 9.0, 2: 1.0}
        policy = EpsilonGreedyPolicy(
            action_space=action_space,
            q_value_fn=lambda state, action: q_values[action],
            epsilon=1.0,
            rng=np.random.default_rng(0),
        )
        actions = {policy.choose_action(state=None) for _ in range(50)}
        assert actions.issubset({0, 1, 2})
        assert len(actions) > 1
