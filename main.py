import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os
import argparse
import torch
import random
import numpy as np

from agents.RandomAgent import RandomAgent
from agents.RandomBellmanAgent import RandomBellmanAgent
from agents.QLearningAgent import QLearningAgent
from agents.FuncApproxLRAgent import FuncApproxLRAgent
from agents.ActorCriticAgent import ActorCriticAgent
from agents.REINFORCEAgent import ReinforceAgent


def run_agent(agent_class, agent_name, env, num_episodes, **agent_params):
    """
    Train, evaluate, and save results for an agent.
    """
    print(f"Running {agent_name} agent...")
    results_dir = agent_params.get("results_dir", f'results/{agent_name}')
    os.makedirs(results_dir, exist_ok=True)
    agent = agent_class(env, **agent_params)

    rewards = agent.train(num_episodes=num_episodes)
    # print(rewards)
    
    rewards_dir = os.path.join(results_dir, "rewards")
    os.makedirs(rewards_dir, exist_ok=True)
    np.savetxt(os.path.join(rewards_dir, "rewards.txt"), rewards)
    
    agent.evaluate()
    agent.plot_rewards()
    agent.record_best_play()

    print(f"Finished running {agent_name} agent.\n")

# Setting the seed to ensure reproducability
def reseed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(agent_name="q_learning", size=4, is_slippery=False, num_episodes=1000):
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), is_slippery=is_slippery, render_mode='rgb_array')

    reseed(695)
    env.reset()

    # Define agents
    agents = {
        "random": {
            "class": RandomAgent,
            "params": {
                "results_dir": 'results/random'
            }
        },
        "random_bellman": {
            "class": RandomBellmanAgent,
            "params": {
                "discount_factor": 0.95,
                "results_dir": 'results/random_bellman'
            }
        },
        "q_learning": {
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
        "func_approx_lr": {
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
        "reinforce": {
            "class": ReinforceAgent,
            "params": {
                "state_dim": env.observation_space.n,
                "action_dim": env.action_space.n,
                "hidden_dim": 16,
                "gamma": 0.99,
                "learning_rate": 0.01,
                "results_dir": 'results/reinforce'
            }
        },
        "actor_critic": {
            "class": ActorCriticAgent,
            "params": {
                "state_dim": env.observation_space.n,
                "action_dim": env.action_space.n,
                "hidden_dim": 16,
                "gamma": 0.99,
                "learning_rate": 0.01,
                "results_dir": 'results/actor_critic'
            }
        }
    }

    if agent_name:
        # Run only the selected agent
        if agent_name in agents:
            agent_config = agents[agent_name]
            run_agent(
                agent_class=agent_config["class"],
                agent_name=agent_name,
                env=env,
                num_episodes=num_episodes,
                **agent_config["params"]
            )
        else:
            print(f"Error: Agent '{agent_name}' not found.")
    else:
        # Run all agents sequentially
        for agent_name, agent_config in agents.items():
            run_agent(
                agent_class=agent_config["class"],
                agent_name=agent_name,
                env=env,
                num_episodes=num_episodes,
                **agent_config["params"]
            )

    # Close the environment
    env.close()

# Run specific agent: python main.py --agent q_learning --size 4 --is_slippery False --num_episodes 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL agents on FrozenLake.")
    parser.add_argument(
        "--agent", 
        type=str, 
        default="q_learning", 
        help="Specify the name of the agent to run (e.g., 'q_learning', 'actor_critic'). Run all agents if not specified."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="Specify the size of the map."
    )
    parser.add_argument(
        "--is_slippery", 
        type=bool, 
        default=False, 
        help="Specify whether the environment should be slippery (True or False)."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Specify the number of episodes for training."
    )
    args = parser.parse_args()
    main(agent_name=args.agent, size=args.size, is_slippery=args.is_slippery, num_episodes=args.num_episodes)
