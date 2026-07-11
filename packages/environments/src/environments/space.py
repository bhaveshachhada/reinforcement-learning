from typing import TypeVar, Generic, List, Union, Tuple

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


class Space(Generic[T]):

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def contains(self, x: T) -> bool:
        return self.__contains__(x)

    def sample(self) -> T:
        raise NotImplementedError


class DiscreteSpace(Space[int]):

    def __init__(self, n: int, start: int, rng: np.random.Generator):
        super().__init__(rng)
        self.n = n
        self.start = start

    def contains(self, x: int) -> bool:
        return self.start <= x < (self.n + self.start)

    def sample(self) -> int:
        return self.start + self.rng.choice(self.n)


class ContinuousSpace(Space[npt.NDArray[np.float64]]):
    def __init__(self, low: npt.ArrayLike, high: npt.ArrayLike, rng: np.random.Generator):
        super().__init__(rng)
        self.low: npt.NDArray[np.float64] = np.asarray(low, dtype=np.float64)
        self.high: npt.NDArray[np.float64] = np.asarray(high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape")
        self.shape: Tuple[int, ...] = self.low.shape

    def contains(self, value: npt.ArrayLike) -> bool:
        value = np.asarray(value, dtype=np.float64)
        return value.shape == self.shape and bool((self.low <= value).all() and (value <= self.high).all())

    def is_bounded(self) -> bool:
        return bool(np.isfinite(self.low).all() and np.isfinite(self.high).all())

    def sample(self) -> npt.NDArray[np.float64]:
        low_bounded = np.isfinite(self.low)
        high_bounded = np.isfinite(self.high)

        both_bounded = low_bounded & high_bounded
        only_low_bounded = low_bounded & ~high_bounded
        only_high_bounded = ~low_bounded & high_bounded
        unbounded = ~low_bounded & ~high_bounded

        value = np.empty(self.shape, dtype=np.float64)
        value[both_bounded] = self.rng.uniform(self.low[both_bounded], self.high[both_bounded])
        value[only_low_bounded] = self.low[only_low_bounded] + self.rng.exponential(size=only_low_bounded.sum())
        value[only_high_bounded] = self.high[only_high_bounded] - self.rng.exponential(size=only_high_bounded.sum())
        value[unbounded] = self.rng.standard_normal(size=unbounded.sum())

        return value