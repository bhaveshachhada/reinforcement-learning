import logging

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track


class FrozenLakeTDAgent:

    def __init__(self,
                 env: gym.Env,
                 n: int = 0,
                 discount_factor: float = 0.1,
                 step_size: float = 0.1,
                 initial_epsilon: float = 0.9,
                 final_epsilon: float = 0.05,
                 epsilon_decay: float = 0.01):
        """
        Initializes the frozen lake TD agent
        :param env: the environment
        :param n: TD param `n` which decides how far in the past to update the values
        :param discount_factor: the discount factor for reward
        :param step_size: the step size parameter to decide how far to see while updating values for current state
        :param initial_epsilon: the initial epsilon (which governs exploration / exploitation)
        :param final_epsilon: the final epsilon
        :param epsilon_decay: the epsilon decay
        """
        self.env = env
        self.n = n
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.epsilon = self.initial_epsilon
        self.q: np.ndarray = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=np.float64)

        self.training_errors: list[float] = list()

    def get_action(self, state: int) -> int:
        """
        Chooses an action for the provided state based on epsilon-greedy policy
        :param state: the state for which action is to be provided
        """

        # Explore with prob. eps
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Exploit with prob. (1 - eps)
        else:
            return np.argmax(self.q[state, :])

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update(self,
               state: int,
               action: int,
               reward: float,
               next_state: int,
               terminated: bool):
        future_q_value = (not terminated) * np.max(self.q[next_state, :])
        temporal_difference = reward + (self.discount_factor * future_q_value) - self.q[state, action]
        self.q[state, action] += self.step_size * temporal_difference

        self.training_errors.append(temporal_difference)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("Frozenlake")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s %(filename)s %(name)s %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(f"logs/frozenlake.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main():
    logger = setup_logger()

    NUM_EPISODES = int(3e5)
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (0.9 * NUM_EPISODES)  # reduce the exploration over time
    final_epsilon = 0.0

    ACTIONS = ['L', 'D', 'R', 'U']

    environment = gym.make(id="FrozenLake-v1",
                           desc=None,
                           map_name="8x8",
                           is_slippery=False,
                           render_mode="rgb_array")

    environment = gym.wrappers.RecordVideo(environment, video_folder="frozenlake-agent",
                                           episode_trigger=lambda x: (x + 1) % 100 == 0)
    environment = gym.wrappers.RecordEpisodeStatistics(environment, buffer_length=NUM_EPISODES)

    agent = FrozenLakeTDAgent(
        environment,
        n=0,
        step_size=0.9,
        discount_factor=0.9,
        initial_epsilon=start_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay=epsilon_decay
    )

    for episode in track(range(NUM_EPISODES), description="Training..."):

        state, _ = environment.reset(seed=episode * np.random.randint(episode + 1))
        done = False

        trace = [state]

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated
            )
            done = (terminated or truncated)
            state = next_state

            trace.append(ACTIONS[action])
            trace.append(next_state)

        agent.decay_epsilon()
        logger.info(f'Episode: {episode}, Eps: {agent.epsilon}, trace: {" ".join(str(s) for s in trace)}')

    # print(f'Episode time taken: {environment.time_queue}')
    # print(f'Episode total rewards: {environment.return_queue}')
    # print(f'Episode lengths: {environment.length_queue}')

    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 5000 episode window
    rolling_length = NUM_EPISODES // 20
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        environment.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        environment.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_errors,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

    # for i in range(4):
    #     values = []
    #     for j in range(4):
    #         values.append(agent.q[i + (4 * j)].max())
    #     print(" ".join(str(val) for val in values))
    # for key in sorted(agent.q.keys()):
    #     print(f"{key}: {agent.q[key]}")


if __name__ == '__main__':
    main()
