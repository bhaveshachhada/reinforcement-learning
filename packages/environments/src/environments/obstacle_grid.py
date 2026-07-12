import os
import sys
from enum import Enum
from typing import Tuple, Iterable
import time

import numpy as np

from packages.environments.src.environments.environment import Environment
from packages.environments.src.environments.space import DiscreteSpace


class Direction(Enum):
    """Agent direction enumeration"""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridEnvironment(Environment[int, int]):
    """
    A configurable terminal-based grid environment for RL agents.

    Args:
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        obstacles (Iterable[Tuple[int, int]]): List of (row, col) positions with obstacles
        start_pos (Tuple[int, int]): Starting position (row, col) for agent
    """

    # Triangle symbols for different directions
    AGENT_SYMBOLS = {
        Direction.UP: "▲",
        Direction.RIGHT: "▶",
        Direction.DOWN: "▼",
        Direction.LEFT: "◀",
    }

    # Fallback ASCII symbols (in case unicode doesn't render)
    AGENT_SYMBOLS_ASCII = {
        Direction.UP: "^",
        Direction.RIGHT: ">",
        Direction.DOWN: "v",
        Direction.LEFT: "<",
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: np.random.Generator,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        obstacles: Iterable[Tuple[int, int]] = None,
        use_unicode: bool = True,
    ):
        self.rows = rows
        self.cols = cols
        self.obstacles = set(obstacles) if obstacles else set()

        # Validate start position
        if not self._is_valid_pos(start_pos):
            raise ValueError(f"Invalid start position {start_pos}")
        if not self._is_valid_pos(goal_pos):
            raise ValueError(f"Invalid goal position {goal_pos}")

        self.agent_pos = start_pos
        self.goal_pos = goal_pos
        self.agent_direction = Direction.RIGHT
        self.use_unicode = use_unicode
        self.step_count = 0
        self.rng = rng

        self.action_space = DiscreteSpace(4, 0, self.rng)
        self.observation_space = DiscreteSpace(self.rows * self.cols, 0, self.rng)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not an obstacle)"""
        row, col = pos
        return (
            0 <= row < self.rows and 0 <= col < self.cols and pos not in self.obstacles
        )

    def _is_position_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds)"""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_colliding_with_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Check if position is colliding with obstacle"""
        return pos in self.obstacles

    def state_from_position(self, pos: Tuple[int, int]) -> int:
        assert len(pos) == 2, "Position must be an iterable of (row, col)"
        row, col = pos
        return row * self.cols + col

    def reset(self, start_pos: Tuple[int, int] = None) -> int:
        """Reset environment to initial state"""
        if start_pos:
            if not self._is_valid_pos(start_pos):
                raise ValueError(f"Invalid start position {start_pos}")
            self.agent_pos = tuple(start_pos)
        else:
            self.agent_pos = (0, 0)

        self.agent_direction = Direction.RIGHT
        self.step_count = 0

        return self.state_from_position(self.agent_pos)

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Execute action and return new position and collision flag

        Actions:
            0: Move Forward (in current direction)
            1: Turn Right (change direction clockwise)
            2: Turn Left (change direction counter-clockwise)
            3: Move Backward

        Returns:
            (new_position, reward, is_terminal) - tuple with new position, reward, is_terminal
        """
        terminal = False
        reward = 0

        if action == 0:  # Move forward
            new_pos = self._move_in_direction(self.agent_pos, self.agent_direction)
        elif action == 1:  # Turn right
            self.agent_direction = Direction((self.agent_direction.value + 1) % 4)
            new_pos = tuple(self.agent_pos)
        elif action == 2:  # Turn left
            self.agent_direction = Direction((self.agent_direction.value - 1) % 4)
            new_pos = tuple(self.agent_pos)
        elif action == 3:  # Move backward
            opposite_dir = Direction((self.agent_direction.value + 2) % 4)
            new_pos = self._move_in_direction(self.agent_pos, opposite_dir)
        else:
            raise ValueError(f"Invalid action {action}")

        # Check if new position is valid
        if self._is_colliding_with_obstacle(new_pos):
            reward = -1
            terminal = True

        elif new_pos == self.goal_pos:
            reward = 1
            terminal = True
        elif self._is_position_in_bounds(new_pos):
            self.agent_pos = new_pos

        self.step_count += 1
        return self.state_from_position(new_pos), reward, terminal

    @staticmethod
    def _move_in_direction(
        pos: Tuple[int, int], direction: Direction
    ) -> Tuple[int, int]:
        """Calculate new position after moving in a direction"""
        row, col = pos

        if direction == Direction.UP:
            row -= 1
        elif direction == Direction.DOWN:
            row += 1
        elif direction == Direction.LEFT:
            col -= 1
        elif direction == Direction.RIGHT:
            col += 1

        return row, col

    def render(self, use_ansi_clear: bool = True) -> None:
        """
        Render the environment to terminal.

        Args:
            use_ansi_clear (bool): Use ANSI codes to clear. Set False if having issues.
        """
        # Clear screen
        self._clear_screen(use_ansi_clear)

        # Print header
        print("\n" + "=" * 40)
        print(f"  RL Grid Environment | Step: {self.step_count}")
        print("=" * 40 + "\n")

        # Draw grid
        self._draw_grid()

        # Print info
        print("\nAgent Info:")
        print(f"  Position: {self.agent_pos}")
        print(f"  Direction: {self.agent_direction.name}")
        print(f"  Obstacles: {len(self.obstacles)}")

    def _clear_screen(self, use_ansi: bool = True) -> None:
        """Clear terminal screen"""
        if use_ansi:
            # ANSI escape codes (works on most terminals)
            sys.stdout.write("\033[2J")  # Clear screen
            sys.stdout.write("\033[H")  # Move cursor to home
            sys.stdout.flush()
        else:
            # Fallback: platform-specific clear command
            os.system("cls" if os.name == "nt" else "clear")

    def _draw_grid(self) -> None:
        """Draw the grid with agent and obstacles"""
        # Select agent symbols based on unicode support
        symbols = self.AGENT_SYMBOLS if self.use_unicode else self.AGENT_SYMBOLS_ASCII

        # Top border
        top_border = "┌" + "─" * (self.cols * 4 - 1) + "┐"
        print(top_border)

        # Grid rows
        for row in range(self.rows):
            row_str = "│"

            for col in range(self.cols):
                cell_content = "   "

                # Check what to draw in this cell
                if (row, col) == tuple(self.agent_pos):
                    # Draw agent with direction indicator
                    symbol = symbols[self.agent_direction]
                    cell_content = " " + symbol + " "
                elif (row, col) in self.obstacles:
                    # Draw obstacle
                    cell_content = " # "

                row_str += cell_content

                # Add separator between columns (except after last column)
                if col < self.cols - 1:
                    row_str += "│"

            row_str += "│"
            print(row_str)

            # Add horizontal separator between rows (except after last row)
            if row < self.rows - 1:
                sep = "├" + "┼".join(["───" for _ in range(self.cols)]) + "┤"
                print(sep.replace("┼", "┼").replace("├", "├").replace("┤", "┤"))

        # Bottom border
        bottom_border = "└" + "─" * (self.cols * 4 - 1) + "┘"
        print(bottom_border)

    def get_state(self) -> dict:
        """Get current environment state"""
        return {
            "position": tuple(self.agent_pos),
            "direction": self.agent_direction,
            "grid_shape": (self.rows, self.cols),
            "obstacles": self.obstacles,
            "step": self.step_count,
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def example_basic():
    """Basic example of grid environment"""
    print("\n" + "=" * 50)
    print("  BASIC GRID ENVIRONMENT EXAMPLE")
    print("=" * 50)

    # Create environment: 5x6 grid with obstacles
    obstacles = [(1, 2), (2, 2), (3, 2), (2, 4)]
    env = GridEnvironment(rows=5, cols=6, obstacles=obstacles, start_pos=(0, 0))

    env.render()
    input("\nPress Enter to continue...")

    # Simulate some movements
    actions = [
        0,  # Move forward (right)
        0,  # Move forward (right)
        1,  # Turn right
        0,  # Move forward (down)
        0,  # Move forward (down)
        2,  # Turn left
        0,  # Move forward (right)
    ]

    for action in actions:
        pos, collision = env.step(action)
        env.render()

        if collision:
            print("\n⚠️  Collision detected!")

        time.sleep(1)  # Pause between steps for visibility
        input("Press Enter for next step...")


def example_interactive():
    """Interactive example where you control the agent"""
    print("\n" + "=" * 50)
    print("  INTERACTIVE GRID ENVIRONMENT")
    print("=" * 50)
    print("\nControls:")
    print("  0: Move Forward")
    print("  1: Turn Right")
    print("  2: Turn Left")
    print("  3: Move Backward")
    print("  q: Quit\n")

    # Create a larger environment with more obstacles
    obstacles = [(2, 1), (2, 2), (2, 3), (4, 4), (4, 5), (4, 6), (1, 5)]

    env = GridEnvironment(rows=7, cols=8, obstacles=obstacles, start_pos=(3, 0))

    env.render()

    while True:
        try:
            user_input = input("\nEnter action (0/1/2/3/q): ").strip().lower()

            if user_input == "q":
                print("Exiting...")
                break

            action = int(user_input)
            if action not in [0, 1, 2, 3]:
                print("Invalid action! Use 0, 1, 2, or 3")
                continue

            pos, collision = env.step(action)
            env.render()

            if collision:
                print("\n⚠️  Collision detected! (tried to hit wall/obstacle)")

        except ValueError:
            print("Invalid input! Enter a number (0-3) or 'q'")


def example_with_fallback():
    """Example using ASCII fallback symbols (for terminals without unicode support)"""
    print("\n" + "=" * 50)
    print("  ASCII FALLBACK EXAMPLE (No Unicode)")
    print("=" * 50)

    obstacles = [(2, 2), (3, 3), (1, 4)]
    env = GridEnvironment(
        rows=5,
        cols=6,
        obstacles=obstacles,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        rng=np.random.default_rng(1),
        use_unicode=False,  # Use ASCII symbols
    )

    env.render()

    # Demo sequence
    actions = [0, 0, 1, 0, 2, 0, 0]

    for action in actions:
        env.step(action)
        env.render()
        time.sleep(0.8)


if __name__ == "__main__":
    # import sys

    print("\nSelect an example:")
    print("1. Basic (automated demo)")
    print("2. Interactive (you control)")
    print("3. ASCII Fallback")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        example_basic()
    elif choice == "2":
        example_interactive()
    elif choice == "3":
        example_with_fallback()
    else:
        print("Invalid choice")
