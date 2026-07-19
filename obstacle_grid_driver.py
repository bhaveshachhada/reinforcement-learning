import time
from typing import Tuple

import numpy as np

from packages.agents.src.agents.sarsa import Sarsa
from packages.environments.src.environments.obstacle_grid import (
    Direction,
    GridEnvironment,
    StochasticGridEnvironment,
)
from packages.environments.src.environments.space import DiscreteSpace
from packages.policies.src.policies.policy import EpsilonGreedyPolicy


def example_basic():
    """Basic example of grid environment"""
    print("\n" + "=" * 50)
    print("  BASIC GRID ENVIRONMENT EXAMPLE")
    print("=" * 50)

    # Create environment: 5x6 grid with obstacles
    obstacles = [(1, 2), (2, 2), (3, 2), (2, 4)]
    env = GridEnvironment(
        rows=5,
        cols=6,
        obstacles=obstacles,
        start_pos=(0, 0, Direction.RIGHT.value),
        goal_pos=(4, 4, Direction.UP.value),
        rng=np.random.default_rng(1),
    )

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
        pos, reward, is_terminal = env.step(action)
        env.render()

        if is_terminal:
            print("\n⚠️  Episode finished with reward={}!".format(reward))
            break

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

    env = GridEnvironment(
        rows=7,
        cols=8,
        obstacles=obstacles,
        start_pos=(3, 0, Direction.UP.value),
        goal_pos=(5, 5, Direction.UP.value),
        rng=np.random.default_rng(1),
    )

    env.render()

    valid_inputs = ["w", "d", "a", "s"]

    while True:
        try:
            user_input = input("\nEnter action (0/1/2/3/q): ").strip().lower()

            if user_input == "q":
                print("Exiting...")
                break

            action = user_input.strip().lower()
            if action not in valid_inputs:
                print("Invalid action! Use w, s, a, or d.")
                continue

            action = valid_inputs.index(action)
            pos, reward, is_terminal = env.step(action)
            env.render()

            if is_terminal:
                print("\n⚠️  Episode finished with reward={}!".format(reward))
                break

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
        start_pos=(0, 0, Direction.RIGHT.value),
        goal_pos=(4, 4, Direction.UP.value),
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
    rng = np.random.default_rng(1)
    start_state = (0, 0, 0)
    n_rows = 6
    n_cols = 6
    goal_state = (n_rows - 1, n_cols - 1, 0)
    # environment = GridEnvironment(
    #     rows=n_rows,
    #     cols=n_cols,
    #     rng=rng,
    #     start_pos=start_state,
    #     goal_pos=goal_state,
    #     obstacles=[
    #         # (1, 0),
    #         # (1, 1),
    #         # (1, 2),
    #         # (1, 3),
    #         # (3, 0),
    #         # (3, 2),
    #         # (3, 3),
    #         # (3, 4),
    #     ],
    # )

    environment = StochasticGridEnvironment(
        rows=n_rows,
        cols=n_cols,
        rng=rng,
        start_pos=start_state,
        goal_pos=goal_state,
        obstacles=[
            # (1, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            # (3, 0),
            # (3, 2),
            # (3, 3),
            # (3, 4),
        ],
    )

    # 3x3 cells, 4 possible head directions in each cell, 4 possible actions that the agent can take
    # q_table = rng.normal(loc=0., scale=.5, size=(3, 3, 4, 4))
    q_table = np.zeros(shape=(n_rows, n_cols, 4, 4), dtype=np.float64)
    q_table[goal_state[0], goal_state[1], goal_state[2], :] = np.zeros(
        shape=(1, 1, 1, 4)
    )

    action_space = DiscreteSpace(4, start=0, rng=rng)
    policy: EpsilonGreedyPolicy[Tuple[int, int]] = EpsilonGreedyPolicy(
        action_space=action_space,
        epsilon=0.1,
        rng=rng,
        q_value_fn=lambda s, a: float(q_table[s[0], s[1], s[2], a]),
    )

    def set_q_value(s, a, value):
        q_table[s[0], s[1], s[2], a] = value

    agent: Sarsa[Tuple[int, int, int], int] = Sarsa(
        env=environment,
        policy=policy,
        q_value_getter=lambda s, a: float(q_table[s[0], s[1], s[2], a]),
        q_value_setter=set_q_value,
        discount_rate=0.99,
        lr=0.11,
        rng=rng,
    )
    # agent: QLearningAgent[Tuple[int, int, int], int] = QLearningAgent(
    #     env=environment,
    #     policy=policy,
    #     q_value_getter=lambda s, a: float(q_table[s[0], s[1], s[2], a]),
    #     q_value_setter=set_q_value,
    #     discount_factor=0.95,
    #     learning_rate=0.01,
    # )

    steps = []
    for episode in range(1000):
        print("\033[36m ======== Episode {} started =======\033[0m".format(episode))

        step = 0

        reward, terminated = 0, False

        state = start_state
        action = agent.choose_action(state=state)
        while not terminated:
            print(f"\033[32m[training loop]\033[0m: started in {state=}, {action=}")
            next_state, reward, terminated = environment.step(action=action)
            print(
                f"\033[32m[training loop]\033[0m env step done, {state=}, {action=}, {reward=}, {next_state=}, {terminated=}"
            )
            next_action = None if terminated else agent.choose_action(next_state)
            agent.step(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                next_action=next_action,
                done=terminated,
            )
            print("\033[32m[training loop]\033[0m agent q_value update done")
            state = next_state
            action = next_action
            step += 1
            print(f"\033[32m[training loop]\033[0m: ended in {state=}")

        steps.append(step)
        environment.reset()
        print(
            "\033[36m ======== Episode {} ended, {} steps =======\033[0m".format(
                episode, step
            )
        )
        print("\n\n")

    print(
        f"Training completed, with {sum(steps) / len(steps):.2f} average steps per episode\n\n"
    )

    print(q_table.mean(axis=(2, 3)))

    input("Training completed, Press Enter to continue...")

    done = False
    state = environment.reset()
    policy.epsilon = 0.0
    environment.render()
    time.sleep(0.4)
    while not done:
        action = agent.choose_action(state=state)
        next_state, reward, done = environment.step(action)
        environment.render()
        time.sleep(0.4)
        state = next_state

    # print("\nSelect an example:")
    # print("1. Basic (automated demo)")
    # print("2. Interactive (you control)")
    # print("3. ASCII Fallback")
    #
    # choice = input("\nEnter choice (1/2/3): ").strip()
    #
    # if choice == "1":
    #     example_basic()
    # elif choice == "2":
    #     example_interactive()
    # elif choice == "3":
    #     example_with_fallback()
    # else:
    #     print("Invalid choice")
