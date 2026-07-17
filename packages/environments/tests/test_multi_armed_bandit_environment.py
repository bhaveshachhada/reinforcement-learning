import numpy as np
import pytest

from packages.environments.src.environments.multi_armed_bandit import (
    MultiArmedBanditEnvironment,
)


class TestMultiArmedBanditEnvironment:
    def test_init_sets_up_action_space_for_each_arm(self):
        env = MultiArmedBanditEnvironment(n_arms=5, rng=np.random.default_rng(0))
        assert env.n_arms == 5
        assert env.action_space.n == 5
        assert env.action_space.start == 0
        assert env.reward_mean_values.shape == (5,)

    def test_observation_space_is_none(self):
        env = MultiArmedBanditEnvironment(n_arms=3, rng=np.random.default_rng(0))
        assert env.observation_space is None

    def test_step_returns_no_observation_and_not_terminal(self):
        env = MultiArmedBanditEnvironment(n_arms=3, rng=np.random.default_rng(0))
        observation, _, terminal = env.step(action=0)
        assert observation is None
        assert terminal is False

    def test_step_reward_is_mean_value_plus_rng_noise(self):
        rng = np.random.default_rng(0)
        env = MultiArmedBanditEnvironment(n_arms=3, rng=rng)

        predictor = np.random.default_rng()
        predictor.bit_generator.state = rng.bit_generator.state
        expected_noise = predictor.normal(loc=0, scale=3)

        _, reward, _ = env.step(action=1)

        assert reward == pytest.approx(env.reward_mean_values[1] + expected_noise)

    def test_reset_is_not_implemented(self):
        env = MultiArmedBanditEnvironment(n_arms=3, rng=np.random.default_rng(0))
        with pytest.raises(NotImplementedError):
            env.reset()

    def test_render_is_not_implemented(self):
        env = MultiArmedBanditEnvironment(n_arms=3, rng=np.random.default_rng(0))
        with pytest.raises(NotImplementedError):
            env.render()
