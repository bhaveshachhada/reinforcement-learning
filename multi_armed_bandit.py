import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track


def argmax(values: np.array):
    """
    Argmax function with random tie breaking
    :param values: np.array
    :return: same as argmax, but when there is tie, gives random among all indices with max value
    """
    return np.random.choice(np.flatnonzero(values == values.max()))


class MultiArmBanditEnvironment:

    def __init__(self, n_arms: int):
        self.n_arms: int = n_arms

        # random reward mean values with mean 0, variance 1
        self.reward_mean_values = 2*np.random.randn(self.n_arms)

    def init_environment(self):
        # random reward mean values with mean 0, variance 1
        self.reward_mean_values = 2*np.random.randn(self.n_arms)

    def env_step(self, action: int):
        # variance = 1, mean = reward_mean_values[index]
        return self.reward_mean_values[action] + np.random.randn()


class Agent:

    def __init__(self, n_arms: int, epsilon: float):
        self.n_arms: int = n_arms
        self.q_values = np.zeros(self.n_arms)
        self.arm_pull_count = np.zeros(self.n_arms, dtype=np.uint64)
        self.epsilon: float = epsilon

    def agent_step(self, previous_action: int, previous_reward: float) -> int:
        raise NotImplementedError


class EpsilonGreedyAgent(Agent):

    def __init__(self, n_arms: int, epsilon: float):
        Agent.__init__(self, n_arms, epsilon)

    def agent_step(self, previous_action: int, previous_reward: float) -> int:
        self.arm_pull_count[previous_action] += 1
        old_q_value = self.q_values[previous_action]
        new_q_value = old_q_value + ((1 / self.arm_pull_count[previous_action]) * (previous_reward - old_q_value))
        self.q_values[previous_action] = new_q_value
        return np.random.randint(self.n_arms) if np.random.random() < self.epsilon else argmax(self.q_values)


if __name__ == '__main__':

    n_runs = 2000
    steps_per_run = 1000

    n_arms = 10

    average_best = 0
    greedy_agent_rewards = np.zeros((n_runs, steps_per_run))
    epsilon_greedy_agent_rewards = np.zeros((n_runs, steps_per_run))
    for run in track(range(n_runs)):

        np.random.seed(run)

        environment = MultiArmBanditEnvironment(n_arms=n_arms)
        average_best += np.max(environment.reward_mean_values)

        # ~~~~~~~~~~~~~ GREEDY AGENT ~~~~~~~~~~~~~ #
        # greedy agent is special case of epsilon greedy with epsilon = 0
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.0)

        selected_action, reward = 0, 0
        for step in range(steps_per_run):
            selected_action = agent.agent_step(selected_action, reward)
            reward = environment.env_step(selected_action)
            greedy_agent_rewards[run, step] = reward

        # ~~~~~~~~~~~~~ EPSILON GREEDY AGENT ~~~~~~~~~~~~~ #
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1)

        selected_action, reward = 0, 0
        for step in range(steps_per_run):
            selected_action = agent.agent_step(selected_action, reward)
            reward = environment.env_step(selected_action)
            epsilon_greedy_agent_rewards[run, step] = reward

    greedy_scores = np.mean(greedy_agent_rewards, axis=0)
    epsilon_greedy_scores = np.mean(epsilon_greedy_agent_rewards, axis=0)
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot([average_best / n_runs for _ in range(steps_per_run)], linestyle="--")
    plt.plot(greedy_scores)
    plt.plot(epsilon_greedy_scores)
    plt.legend(["Best Possible", "Greedy", "Epsilon greedy"])
    plt.title("Average Reward of Greedy Agent vs Epsilon Greedy")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
