import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os

from agents.RandomAgent import RandomAgent
from agents.RandomBellmanAgent import RandomBellmanAgent
from agents.QLearningAgent import QLearningAgent
from agents.FuncApproxLRAgent import FuncApproxLRAgent
from agents.ActorCriticAgent import ActorCriticAgent

def run_agent(agent_class, agent_name, env, **agent_params):
    """
    Train, evaluate, and save results for an agent.
    """
    print(f"Running {agent_name} agent...")
    results_dir = agent_params.get("results_dir", f'results/{agent_name}')
    os.makedirs(results_dir, exist_ok=True)
    agent = agent_class(env, **agent_params)

    rewards = agent.train(num_episodes=10000)
    agent.evaluate()
    agent.plot_rewards()
    agent.record_best_play()

    print(f"Finished running {agent_name} agent.\n")

def main():
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=True, render_mode='rgb_array')

    agents = [
        {
            "name": "random",
            "class": RandomAgent,
            "params": {
                "results_dir": 'results/random'
            }
        },
        {
            "name": "random_bellman",
            "class": RandomBellmanAgent,
            "params": {
                "discount_factor": 0.95,
                "results_dir": 'results/random_bellman'
            }
        },
        {
            "name": "q_learning",
            "class": QLearningAgent,
            "params": {
                "learning_rate": 0.8,
                "discount_factor": 0.95,
                "epsilon": 1.0,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.01,
                "results_dir": 'results/q_learning'
            }
        },
        {
            "name": "func_approx_lr",
            "class": FuncApproxLRAgent,
            "params": {
                "learning_rate": 0.01,
                "discount_factor": 0.95,
                "epsilon": 1.0,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.01,
                "results_dir": 'results/func_approx_lr'
            }
        },
        {
            "name": "actor_critic",
            "class": ActorCriticAgent,
            "params": {
                "learning_rate": 0.01,
                "gamma": 0.99,
                "results_dir": 'results/actor_critic'
            }
        }
    ]

    # Run each agent sequentially
    for agent_config in agents:
        run_agent(
            agent_class=agent_config["class"],
            agent_name=agent_config["name"],
            env=env,
            **agent_config["params"]
        )

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()