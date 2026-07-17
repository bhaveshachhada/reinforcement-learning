import numpy as np
import pytest

from packages.environments.src.environments.space import (
    ContinuousSpace,
    DiscreteSpace,
)


class TestDiscreteSpace:
    def test_contains_returns_true_for_value_in_range(self):
        space = DiscreteSpace(n=5, start=2, rng=np.random.default_rng(0))
        assert space.contains(2) is True
        assert space.contains(6) is True

    def test_contains_returns_false_for_value_below_start(self):
        space = DiscreteSpace(n=5, start=2, rng=np.random.default_rng(0))
        assert space.contains(1) is False

    def test_contains_returns_false_for_value_at_or_above_end(self):
        space = DiscreteSpace(n=5, start=2, rng=np.random.default_rng(0))
        assert space.contains(7) is False

    def test_sample_is_within_bounds(self):
        space = DiscreteSpace(n=5, start=2, rng=np.random.default_rng(0))
        for _ in range(100):
            assert space.contains(space.sample())

    def test_sample_is_offset_by_start(self):
        rng = np.random.default_rng(42)
        space = DiscreteSpace(n=3, start=10, rng=rng)
        sample = space.sample()
        assert 10 <= sample < 13


class TestContinuousSpace:
    def test_raises_value_error_when_low_high_shapes_differ(self):
        with pytest.raises(ValueError):
            ContinuousSpace(
                low=[0.0, 0.0], high=[1.0, 1.0, 1.0], rng=np.random.default_rng(0)
            )

    def test_contains_true_for_value_within_bounds(self):
        space = ContinuousSpace(
            low=[0.0, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        assert space.contains([0.5, 0.5]) is True

    def test_contains_false_for_value_outside_bounds(self):
        space = ContinuousSpace(
            low=[0.0, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        assert space.contains([1.5, 0.5]) is False

    def test_contains_false_for_mismatched_shape(self):
        space = ContinuousSpace(
            low=[0.0, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        assert space.contains([0.5, 0.5, 0.5]) is False

    def test_is_bounded_true_when_all_finite(self):
        space = ContinuousSpace(
            low=[0.0, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        assert space.is_bounded() is True

    def test_is_bounded_false_when_low_is_infinite(self):
        space = ContinuousSpace(
            low=[-np.inf, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        assert space.is_bounded() is False

    def test_sample_bounded_dimension_stays_within_bounds(self):
        space = ContinuousSpace(
            low=[0.0, 0.0], high=[1.0, 1.0], rng=np.random.default_rng(0)
        )
        for _ in range(100):
            sample = space.sample()
            assert space.contains(sample)

    def test_sample_only_low_bounded_dimension_is_at_least_low(self):
        space = ContinuousSpace(low=[0.0], high=[np.inf], rng=np.random.default_rng(0))
        for _ in range(100):
            assert space.sample()[0] >= 0.0

    def test_sample_only_high_bounded_dimension_is_at_most_high(self):
        space = ContinuousSpace(low=[-np.inf], high=[0.0], rng=np.random.default_rng(0))
        for _ in range(100):
            assert space.sample()[0] <= 0.0

    def test_sample_unbounded_dimension_has_correct_shape(self):
        space = ContinuousSpace(
            low=[-np.inf, -np.inf],
            high=[np.inf, np.inf],
            rng=np.random.default_rng(0),
        )
        sample = space.sample()
        assert sample.shape == (2,)
