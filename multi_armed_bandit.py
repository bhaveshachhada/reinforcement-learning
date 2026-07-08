import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rich.progress import track


def argmax(values: NDArray, rng: np.random.Generator) -> np.uint8:
    """
    Argmax function with random tie breaking
    :param values: np.array
    :return: same as argmax, but when there is tie, gives random among all indices with max value
    """
    return rng.choice(np.flatnonzero(values == values.max()))


class MultiArmBanditEnvironment:

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms: int = n_arms
        self.rng = rng

        # random reward mean values with mean 0, variance 1
        self.reward_mean_values = self.rng.normal(loc=20, scale=5, size=(n_arms,))

    def env_step(self, action: int):
        return self.reward_mean_values[action] + self.rng.normal(loc=0, scale=3)


class EpsilonGreedyAgent:

    def __init__(self, n_arms: int, epsilon: float, rng: np.random.Generator):
        self.n_arms: int = n_arms
        self.q_values = np.zeros(self.n_arms)
        self.arm_pull_count: np.ndarray[tuple[()], np.uint8] = np.zeros(self.n_arms, dtype=np.uint64)
        self.epsilon = epsilon
        self.rng = rng

    def agent_step(self, previous_action: int, previous_reward: float) -> int:
        old_q_value = self.q_values[previous_action]
        step_size = 1 / max(1, self.arm_pull_count[previous_action])
        new_q_value = old_q_value + (step_size * (previous_reward - old_q_value))
        self.arm_pull_count[previous_action] += 1
        self.q_values[previous_action] = new_q_value
        return self.rng.integers(low=0, high=self.n_arms, dtype=np.uint8) if self.rng.binomial(n=1, p=self.epsilon) else argmax(self.q_values, self.rng)


def run_agent(env: MultiArmBanditEnvironment, agent_: EpsilonGreedyAgent, steps: int):
    rewards = np.zeros((steps,))
    selected_action, reward = 0, 0
    for step in range(steps):
        selected_action = agent_.agent_step(selected_action, reward)
        reward = env.env_step(selected_action)
        rewards[step] = reward
    return rewards


if __name__ == '__main__':

    n_runs = 2000
    steps_per_run = 1000

    n_arms = 10

    average_best = 0
    greedy_agent_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_0_1_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_0_3_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_0_5_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_0_7_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_1_0_rewards = np.zeros((n_runs, steps_per_run))

    timer_start = time.time()
    for run in track(range(n_runs)):

        rng = np.random.default_rng(run)

        environment = MultiArmBanditEnvironment(n_arms=n_arms, rng=rng)
        average_best += np.max(environment.reward_mean_values)

        # ~~~~~~~~~~~~~ GREEDY AGENT ~~~~~~~~~~~~~ #
        # greedy agent is special case of epsilon greedy with epsilon = 0
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.0, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        greedy_agent_rewards[run, :] = rewards

        # ~~~~~~~~~~~~~ EPSILON GREEDY AGENTS ~~~~~~~~~~~~~ #
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_1_rewards[run, :] = rewards

        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.3, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_3_rewards[run, :] = rewards

        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.5, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_5_rewards[run, :] = rewards

        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.7, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_7_rewards[run, :] = rewards

        # epsilon = 1 means complete random walk amongst actions
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=1, rng=rng)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_1_0_rewards[run, :] = rewards


    timer_end = time.time()
    duration = round(timer_end - timer_start, 4)
    print(f"Took {duration:.2f} seconds for {n_runs} runs, {duration / n_runs:.2f}s / run")

    agent_name_score_mapping = [
        ("Greedy", greedy_agent_rewards),
        ("EpsilonGreedy(0.1)", epsilon_greedy_0_1_rewards),
        ("EpsilonGreedy(0.3)", epsilon_greedy_0_3_rewards),
        ("EpsilonGreedy(0.5)", epsilon_greedy_0_5_rewards),
        ("EpsilonGreedy(0.7)", epsilon_greedy_0_7_rewards),
        ("EpsilonGreedy(1.0)", epsilon_greedy_1_0_rewards),
    ]

    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot([average_best / n_runs for _ in range(steps_per_run)], linestyle="--")
    legends = ["Best Possible"]
    for agent_type, scores in agent_name_score_mapping:
        plt.plot(np.mean(scores, axis=0))
        legends.append(agent_type)

    plt.legend(legends)
    plt.title("Average Reward of Greedy Agent vs Epsilon Greedy")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
