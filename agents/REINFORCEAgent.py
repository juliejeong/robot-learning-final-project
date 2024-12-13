import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# import tensorflow as tf
# from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Similar to A4
class PolicyNet(nn.Module):
  def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
    """Policy network for the REINFORCE algorithm.

    Args:
      state_dim (int): Dimension of the state space.
      action_dim (int): Dimension of the action space.
      hidden_dim (int): Dimension of the hidden layers.
    """
    super(PolicyNet, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, action_dim)

  def forward(self, state: torch.Tensor):
    """Forward pass of the policy network.

    Args:
      state (torch.Tensor): State of the environment.

    Returns:
      x (torch.Tensor): Probabilities of the actions.
    """
    # print(f"Input state shape: {state.shape}")  # Debug the input shape
    x = torch.nn.functional.relu(self.fc1(state))
    x = self.fc2(x)
    return torch.nn.functional.softmax(x, dim=-1)

class PolicyGradient:
  def __init__(self, env, policy_net, gamma=0.99, learning_rate=0.001, reward_to_go=False):
    self.env = env
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.policy_net = policy_net.to(self.device)
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.reward_to_go = reward_to_go

    torch.manual_seed(0)
    np.random.seed(0)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

  def select_action(self, state):
    # state = torch.tensor(state).float().unsqueeze(0).to(self.device)
    # probs = self.policy_net(state)
    # dist = torch.distributions.Categorical(probs)
    # action = dist.sample()
    # return action.item()
  
    state_tensor = torch.zeros(self.env.observation_space.n).to(self.device)
    state_tensor[state] = 1.0  # One-hot encode the state
    state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

    probs = self.policy_net(state_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item()

  def compute_loss(self, episode, gamma):
    states, actions, rewards = zip(*episode)
    states = torch.tensor(states).float().to(self.device)

    batch_size = len(states)
    one_hot_states = torch.zeros(batch_size, self.env.observation_space.n).to(self.device)
    one_hot_states[range(batch_size), states.long()] = 1.0 

    actions = torch.tensor(actions).to(self.device)
    rewards = torch.tensor(rewards).to(self.device)

    # Compute discounted rewards
    discounted_rewards = []
    sum_discounted_reward = 0
    for reward in reversed(rewards):
        sum_discounted_reward = reward + gamma * sum_discounted_reward
        discounted_rewards.insert(0, sum_discounted_reward)

    discounted_rewards = torch.tensor(discounted_rewards).float().to(self.device)

    log_probs = torch.log(self.policy_net(one_hot_states))
    log_probs_actions = log_probs[np.arange(len(actions)), actions]
    loss = -torch.sum(log_probs_actions * discounted_rewards)

    return loss

  def update_policy(self, episodes, optimizer, gamma):
    total_loss = 0
    for episode in episodes:
      loss = self.compute_loss(episode, gamma)
      total_loss += loss

    total_loss = total_loss / len(episodes)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

  def run_episode(self):
    state, _ = self.env.reset()
    episode = []
    done = False
    while not done:
      # print(f"State at step: {state}, Shape: {np.array(state).shape}")

      action = self.select_action(state)
      next_state, reward, done, _, _ = self.env.step(action)
      episode.append((state, action, reward))
      state = next_state
    return episode

  def train(self, num_iterations, batch_size, gamma, learning_rate):
    self.policy_net.train()
    optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    avg_rewards = []
    for i in range(num_iterations):
      episodes = [self.run_episode() for _ in range(batch_size)]
      self.update_policy(episodes, optimizer, gamma)
      if i % 10 == 0:
        avg_reward = self.evaluate(10)
        avg_rewards.append(avg_reward)
      if i % 100 == 0:
                print(f"Episode {i}, Average Reward: {np.mean(avg_rewards[-100:]):.2f}")
    return avg_rewards
  
  def evaluate(self, num_episodes=100):
    """Evaluate the policy network by running multiple episodes.

    Args:
      num_episodes (int): Number of episodes to run

    Returns:
      average_reward (float): Average total reward per episode
    """
    self.policy_net.eval()
    total_reward = 0
    for i in range(num_episodes):
      episode = self.run_episode()
      total_reward += sum(reward for state, action, reward in episode)

    avg_total_reward = total_reward / num_episodes

    return avg_total_reward

class ReinforceAgent:
  def select_action(self, state):
    return self.policy_gradient.select_action(state)
  
  def __init__(self, env, state_dim, action_dim, hidden_dim, gamma=0.99, learning_rate=0.01, reward_to_go=False, results_dir='results/q_learning'):
    # Create results directory
    self.results_dir = results_dir
    os.makedirs(self.results_dir, exist_ok=True)

    self.env = env

    self.gamma = gamma
    self.learning_rate = learning_rate
    
    self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim)
    self.policy_gradient = PolicyGradient(env, self.policy_net, gamma, learning_rate, reward_to_go)
    self.rewards = []

  def train(self, num_episodes):
    self.rewards = self.policy_gradient.train(num_iterations=num_episodes, batch_size=10, gamma=self.gamma, learning_rate=self.learning_rate)
    return self.rewards

  def plot_rewards(self):
    """
    Plot cumulative average rewards and save to results directory
    """
    # Create plots subdirectory
    plots_dir = os.path.join(self.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(self.rewards) / np.arange(1, len(self.rewards) + 1))
    plt.title('REINFORCE Performance on Frozen Lake')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Rewards plot saved to {plot_path}")
  
  def evaluate(self, num_eval_episodes=100):
    """
    Evaluate the trained agent and save results
    """
    self.policy_net.eval()
    total_rewards = 0
    for _ in range(num_eval_episodes):
      state, _ = self.env.reset()
      done = False
      while not done:
        action = self.select_action(state)
        state, reward, done, _ , _= self.env.step(action)
        total_rewards += reward
    
    avg_reward = total_rewards / num_eval_episodes
    print(f"Average Reward over {num_eval_episodes} evaluation episodes: {avg_reward}")
    
    # Save evaluation results
    data_dir = os.path.join(self.results_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'evaluation_results.txt'), 'w') as f:
      f.write(f"Average Reward: {avg_reward}\n")
      f.write(f"Number of Evaluation Episodes: {num_eval_episodes}")
    
    return avg_reward
  
  def record_best_play(self):
    """
    Record the best gameplay episode
    """
    # Create video subdirectory
    video_dir = os.path.join(self.results_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Wrap environment with video recorder
    record_env = gym.wrappers.RecordVideo(
      self.env, 
      video_dir, 
      episode_trigger=lambda episode: True,  # Record all episodes
      name_prefix='best_reinforce_play'
    )
    
    # Reset environment and play the best episode
    state, _ = record_env.reset()
    done = False
    total_reward = 0
    
    while not done:
      # Choose action based on learned policy
      action = self.policy_gradient.select_action(state)
      state, reward, done, _, _ = record_env.step(action)
      total_reward += reward
    
    print(f"Best Play Recorded. Total Reward: {total_reward}")
    print(f"Video saved in {video_dir}")
    record_env.close()
