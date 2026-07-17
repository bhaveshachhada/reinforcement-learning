import numpy as np
import pytest

from packages.environments.src.environments.obstacle_grid import (
    Direction,
    GridEnvironment,
)


def make_env(
    rows=3,
    cols=3,
    start_pos=(1, 1, Direction.RIGHT.value),
    goal_pos=(2, 2, Direction.UP.value),
    obstacles=None,
):
    return GridEnvironment(
        rows=rows,
        cols=cols,
        rng=np.random.default_rng(0),
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
    )


class TestConstructorValidation:
    def test_raises_for_out_of_bounds_start_pos(self):
        with pytest.raises(ValueError):
            make_env(start_pos=(5, 5, 0))

    def test_raises_for_start_pos_on_obstacle(self):
        with pytest.raises(ValueError):
            make_env(start_pos=(1, 1, 0), obstacles=[(1, 1)])

    def test_raises_for_out_of_bounds_goal_pos(self):
        with pytest.raises(ValueError):
            make_env(goal_pos=(5, 5, 0))

    def test_agent_direction_matches_start_pos(self):
        env = make_env(start_pos=(0, 0, Direction.LEFT.value))
        assert env.agent_direction == Direction.LEFT


class TestReset:
    def test_reset_without_start_pos_returns_origin_facing_up(self):
        env = make_env()
        state = env.reset()
        assert state == (0, 0, Direction.UP.value)
        assert env.step_count == 0

    def test_reset_with_valid_start_pos(self):
        env = make_env()
        env.step(0)
        state = env.reset(start_pos=(2, 0, Direction.DOWN.value))
        assert state == (2, 0, Direction.DOWN.value)
        assert env.step_count == 0

    def test_reset_raises_for_invalid_start_pos(self):
        env = make_env()
        with pytest.raises(ValueError):
            env.reset(start_pos=(9, 9, 0))


class TestStateFromPosition:
    def test_computes_row_major_index(self):
        env = make_env(rows=3, cols=4)
        assert env.state_from_position((0, 0)) == 0
        assert env.state_from_position((1, 2)) == 6
        assert env.state_from_position((2, 3)) == 11


class TestMoveInDirection:
    @pytest.mark.parametrize(
        "direction, expected",
        [
            (Direction.UP, (0, 1)),
            (Direction.DOWN, (2, 1)),
            (Direction.LEFT, (1, 0)),
            (Direction.RIGHT, (1, 2)),
        ],
    )
    def test_moves_one_cell_in_given_direction(self, direction, expected):
        assert GridEnvironment._move_in_direction((1, 1), direction) == expected


class TestStepMovement:
    def test_move_forward_into_free_cell_updates_position(self):
        env = make_env(start_pos=(1, 1, Direction.RIGHT.value))
        state, reward, terminal = env.step(0)
        assert state == (1, 2, Direction.RIGHT.value)
        assert reward == 0
        assert terminal is False

    def test_move_forward_out_of_bounds_leaves_position_unchanged(self):
        env = make_env(start_pos=(0, 0, Direction.UP.value))
        state, reward, terminal = env.step(0)
        assert state == (0, 0, Direction.UP.value)
        assert reward == 0
        assert terminal is False

    def test_move_backward_moves_opposite_of_facing_direction(self):
        env = make_env(start_pos=(1, 1, Direction.RIGHT.value))
        state, reward, terminal = env.step(3)
        assert state == (1, 0, Direction.RIGHT.value)
        assert reward == 0
        assert terminal is False

    def test_turn_right_changes_direction_but_not_position(self):
        env = make_env(start_pos=(1, 1, Direction.UP.value))
        state, reward, terminal = env.step(1)
        assert state == (1, 1, Direction.RIGHT.value)
        assert reward == 0
        assert terminal is False

    def test_turn_left_wraps_around(self):
        env = make_env(start_pos=(1, 1, Direction.UP.value))
        state, reward, terminal = env.step(2)
        assert state == (1, 1, Direction.LEFT.value)

    def test_invalid_action_raises_value_error(self):
        env = make_env()
        with pytest.raises(ValueError):
            env.step(4)

    def test_step_count_increments_on_every_step(self):
        env = make_env()
        env.step(1)
        env.step(1)
        assert env.step_count == 2


class TestStepCollisionAndGoal:
    def test_moving_into_obstacle_ends_episode_without_moving_there(self):
        env = make_env(start_pos=(1, 1, Direction.RIGHT.value), obstacles=[(1, 2)])
        state, reward, terminal = env.step(0)
        assert reward == -1
        assert terminal is True
        assert state == (1, 1, Direction.RIGHT.value)

    def test_reaching_goal_with_matching_direction_gives_positive_reward(self):
        env = make_env(
            start_pos=(0, 0, Direction.RIGHT.value),
            goal_pos=(0, 1, Direction.RIGHT.value),
        )
        state, reward, terminal = env.step(0)
        assert reward == 1
        assert terminal is True
        assert state == (0, 1, Direction.RIGHT.value)

    def test_reaching_goal_cell_with_wrong_direction_does_not_terminate(self):
        env = make_env(
            start_pos=(0, 0, Direction.RIGHT.value),
            goal_pos=(0, 1, Direction.UP.value),
        )
        state, reward, terminal = env.step(0)
        assert reward == 0
        assert terminal is False
        assert state == (0, 1, Direction.RIGHT.value)

    def test_turning_on_goal_cell_does_not_terminate(self):
        env = make_env(
            start_pos=(2, 2, Direction.UP.value),
            goal_pos=(2, 2, Direction.RIGHT.value),
        )
        state, reward, terminal = env.step(1)  # turn right: UP -> RIGHT
        assert reward == 0
        assert terminal is False
        assert state == (2, 2, Direction.RIGHT.value)


class TestGetState:
    def test_returns_current_position_direction_and_grid_info(self):
        env = make_env(
            rows=3, cols=3, start_pos=(1, 1, Direction.RIGHT.value), obstacles=[(0, 0)]
        )
        env.step(0)  # move forward
        state = env.get_state()
        assert state["position"] == (1, 2)
        assert state["direction"] == Direction.RIGHT
        assert state["grid_shape"] == (3, 3)
        assert state["obstacles"] == {(0, 0)}
        assert state["step"] == 1


class TestRender:
    def test_render_does_not_raise(self, capsys):
        env = make_env()
        env.render(use_ansi_clear=False)
        captured = capsys.readouterr()
        assert "RL Grid Environment" in captured.out
