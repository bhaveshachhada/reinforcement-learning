import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rich.progress import track

from packages.agents.src.agents.agent import Agent
from packages.environments.src.environments.multi_armed_bandit import (
    MultiArmedBanditEnvironment,
)
from packages.environments.src.environments.space import DiscreteSpace
from packages.policies.src.policies.policy import (
    Policy,
    GreedyPolicy,
    EpsilonGreedyPolicy,
)


def argmax(values: NDArray, rng: np.random.Generator) -> np.uint8:
    """
    Argmax function with random tie breaking
    :param values: np.array
    :return: same as argmax, but when there is tie, gives random among all indices with max value
    """
    return rng.choice(np.flatnonzero(values == values.max()))


class MultiArmedBanditAgent(Agent[None, int]):
    def __init__(
        self,
        env: MultiArmedBanditEnvironment,
        policy: Policy,
        q_values: np.ndarray[tuple[int], np.float64],
    ):
        self.env = env
        self.policy = policy

        self.n_arms = env.n_arms
        self.q_values = q_values
        self.arm_pull_count: np.ndarray[tuple[int], np.uint8] = np.zeros(
            self.n_arms, dtype=np.uint64
        )
        self.rng = rng

    def choose_action(self, state: None) -> int:
        return self.policy.choose_action(state)

    def step(self, action: int, reward: Union[int, float]):
        old_q_value = self.q_values[action]
        step_size = 1 / max(1, self.arm_pull_count[action])
        new_q_value = old_q_value + (step_size * (reward - old_q_value))
        self.arm_pull_count[action] += 1
        self.q_values[action] = new_q_value


def run_agent(env: MultiArmedBanditEnvironment, agent_: Agent, steps: int):
    rewards = np.zeros((steps,))
    selected_action, reward = 0, 0
    for step in range(steps):
        agent_.step(selected_action, reward)
        selected_action = agent_.choose_action(None)
        _, reward, _ = env.step(selected_action)
        rewards[step] = reward
    return rewards


if __name__ == "__main__":
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
        # logger.info("Running {} run...".format(run))

        rng = np.random.default_rng(run)

        environment = MultiArmedBanditEnvironment(n_arms=n_arms, rng=rng)
        average_best += np.max(environment.reward_mean_values)

        # common objects
        action_space = DiscreteSpace(n_arms, start=0, rng=rng)

        # ~~~~~~~~~~~~~ GREEDY AGENT ~~~~~~~~~~~~~ #
        # greedy agent is special case of epsilon greedy with epsilon = 0
        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.0, rng=rng)
        # logger.info("Running greedy policy")
        q_values = np.zeros((n_arms,), dtype=np.float64)
        greedy_policy = GreedyPolicy(
            action_space, q_value_fn=lambda state, action: q_values[action], rng=rng
        )
        agent = MultiArmedBanditAgent(environment, greedy_policy, q_values)
        rewards_ = run_agent(environment, agent, steps_per_run)
        greedy_agent_rewards[run, :] = rewards_

        # ~~~~~~~~~~~~~ EPSILON GREEDY AGENTS ~~~~~~~~~~~~~ #
        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1, rng=rng)
        # logger.info("Running epsilon(0.1)")
        q_values = np.zeros((n_arms,), dtype=np.float64)
        epsilon_greedy_policy_0_1 = EpsilonGreedyPolicy(
            action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
            epsilon=0.1,
        )
        agent = MultiArmedBanditAgent(environment, epsilon_greedy_policy_0_1, q_values)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_1_rewards[run, :] = rewards

        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.3, rng=rng)
        # logger.info("Running epsilon(0.3)")
        q_values = np.zeros((n_arms,), dtype=np.float64)
        epsilon_greedy_policy_0_3 = EpsilonGreedyPolicy(
            action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
            epsilon=0.3,
        )
        agent = MultiArmedBanditAgent(environment, epsilon_greedy_policy_0_3, q_values)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_3_rewards[run, :] = rewards

        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.5, rng=rng)
        # logger.info("Running epsilon(0.5)")
        q_values = np.zeros((n_arms,), dtype=np.float64)
        epsilon_greedy_policy_0_5 = EpsilonGreedyPolicy(
            action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
            epsilon=0.5,
        )
        agent = MultiArmedBanditAgent(environment, epsilon_greedy_policy_0_5, q_values)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_5_rewards[run, :] = rewards

        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.7, rng=rng)
        # logger.info("Running epsilon(0.7)")
        q_values = np.zeros((n_arms,), dtype=np.float64)
        epsilon_greedy_policy_0_7 = EpsilonGreedyPolicy(
            action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
            epsilon=0.7,
        )
        agent = MultiArmedBanditAgent(environment, epsilon_greedy_policy_0_7, q_values)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_0_7_rewards[run, :] = rewards

        # epsilon = 1 means complete random walk amongst actions
        # agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=1, rng=rng)
        # logger.info("Running epsilon(1.0)")
        q_values = np.zeros((n_runs,), dtype=np.float64)
        epsilon_greedy_policy_1_0 = EpsilonGreedyPolicy(
            action_space,
            q_value_fn=lambda state, action: q_values[action],
            rng=rng,
            epsilon=1.0,
        )
        agent = MultiArmedBanditAgent(environment, epsilon_greedy_policy_1_0, q_values)
        rewards = run_agent(environment, agent, steps_per_run)
        epsilon_greedy_1_0_rewards[run, :] = rewards

    timer_end = time.time()
    duration = round(timer_end - timer_start, 4)
    print(
        f"Took {duration:.2f} seconds for {n_runs} runs, {duration / n_runs:.2f}s / run"
    )

    agent_name_score_mapping = [
        ("Greedy", greedy_agent_rewards),
        ("EpsilonGreedy(0.1)", epsilon_greedy_0_1_rewards),
        ("EpsilonGreedy(0.3)", epsilon_greedy_0_3_rewards),
        ("EpsilonGreedy(0.5)", epsilon_greedy_0_5_rewards),
        ("EpsilonGreedy(0.7)", epsilon_greedy_0_7_rewards),
        ("EpsilonGreedy(1.0)", epsilon_greedy_1_0_rewards),
    ]

    plt.figure(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")
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
