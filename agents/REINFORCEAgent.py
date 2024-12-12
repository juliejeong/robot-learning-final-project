import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
import gym
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
    x = torch.nn.functional.relu(self.fc1(state))
    x = self.fc2(x)
    return torch.nn.functional.softmax(x, dim=-1)

class PolicyGradient:
  def __init__(self, env, policy_net, seed, gamma=0.99, lr=0.001, reward_to_go=False):
    self.env = env
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.policy_net = policy_net.to(self.device)
    self.gamma = gamma
    self.lr = lr
    self.reward_to_go = reward_to_go
    self.seed = seed

    self.env.seed(self.seed)
    self.env.action_space.seed(self.seed)
    self.env.observation_space.seed(self.seed)

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

  def select_action(self, state):
    state = torch.tensor(state).float().to(self.device)
    probs = self.policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item()

  def compute_loss(self, episode):
    states, actions, rewards = zip(*episode)
    states = torch.tensor(states).float().to(self.device)
    actions = torch.tensor(actions).to(self.device)
    rewards = torch.tensor(rewards).to(self.device)

    discounted_rewards = []
    if not self.reward_to_go:
      sum_discounted_reward = 0
      for reward in reversed(rewards):
        sum_discounted_reward = reward + self.gamma * sum_discounted_reward
        discounted_rewards.insert(0, sum_discounted_reward)
      discounted_rewards = len(rewards) * [discounted_rewards[0]]
    else:
      for t in range(len(rewards)):
        sum_discounted_reward = 0
        for t_prime, reward in enumerate(rewards[t:]):
          sum_discounted_reward += reward * (self.gamma ** t_prime)
        discounted_rewards.append(sum_discounted_reward)

    discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
    log_probs = torch.log(self.policy_net(states))
    log_probs_actions = log_probs[np.arange(len(actions)), actions]
    loss = -torch.sum(log_probs_actions * discounted_rewards)
    return loss

  def update_policy(self, episodes):
    total_loss = 0
    for episode in episodes:
      loss = self.compute_loss(episode)
      total_loss += loss

    total_loss = total_loss / len(episodes)
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()

  def run_episode(self):
    state = self.env.reset()
    episode = []
    done = False
    while not done:
      action = self.select_action(state)
      next_state, reward, done, _ = self.env.step(action)
      episode.append((state, action, reward))
      state = next_state
    return episode

  def train(self, num_iterations, batch_size):
    self.policy_net.train()
    avg_rewards = []
    for i in range(num_iterations):
      episodes = [self.run_episode() for _ in range(batch_size)]
      self.update_policy(episodes)
      if i % 10 == 0:
        avg_reward = self.evaluate(10)
        avg_rewards.append(avg_reward)
    return avg_rewards

'''
# FOR RUNNING THE ALGORITHM -- partially done in main?

# Feel free to use the space below to run experiments and create plots used in your writeup.
env = gym.make("CartPole-v1")
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n, 128)

reinforce = PolicyGradient(env, policy_net, seed, reward_to_go=False)
avg_rewards = reinforce.train(num_iterations=200, batch_size=10, gamma=0.99, lr=0.001)

visualize(algorithm=reinforce, video_name="reinforce")
'''

class ReinforceAgent:
  def __init__(self, env, state_dim, action_dim, hidden_dim, seed, gamma=0.99, lr=0.001, reward_to_go=False):
    self.env = env
    self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim)
    self.policy_gradient = PolicyGradient(env, self.policy_net, seed, gamma, lr, reward_to_go)
    self.rewards = []

  def train(self, num_episodes):
    self.rewards = self.policy_gradient.train(num_iterations=num_episodes, batch_size=10)
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
    plt.plot(np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1))
    plt.title('REINFROCE Performance on Frozen Lake')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Rewards plot saved to {plot_path}")
  
  def evaluate(self, num_episodes=100):
    self.policy_net.eval()
    total_reward = 0
    for _ in range(num_episodes):
      episode = self.run_episode()
      total_reward += sum(reward for _, _, reward in episode)
    avg_total_reward = total_reward / num_episodes
    return avg_total_reward

  def record_best_play(self):
    state = self.env.reset()
    done = False
    trajectory = []
    while not done:
      action = self.select_action(state)
      next_state, reward, done, _ = self.env.step(action)
      trajectory.append((state, action, reward))
      state = next_state
    return trajectory