import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.REINFORCEAgent import PolicyGradient, PolicyNet

class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Value network for the Actor-Critic algorithm.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass of the value network.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            torch.Tensor: Estimated value of the state.
        """
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x
    
class ActorCriticPolicyGradient(PolicyGradient):
    def __init__(self, env, policy_net, value_net):
        """
        Initialize Actor-Critic Agent
        
        Args:
            env (gym.Env): Gymnasium environment
            learning_rate (float): Learning rate for optimization
            gamma (float): Discount factor for future rewards
            results_dir (str): Directory to save results
        """
        
        self.env = env
        # self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)

        torch.manual_seed(0)
        np.random.seed(0)
        
   
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

        # Compute advantages
        values = self.value_net(one_hot_states).squeeze()  # Predicted state values
        advantages = discounted_rewards - values

        # Compute policy loss
        log_probs = torch.log(self.policy_net(one_hot_states))
        log_probs_actions = log_probs[np.arange(len(actions)), actions]
        policy_loss = -torch.sum(log_probs_actions * advantages) / len(rewards)

        # Compute value loss
        value_loss = torch.sum(advantages ** 2) / len(rewards)

        return policy_loss, value_loss

    
    def update_policy(self, episodes, optimizer, value_optimizer, gamma):
        policy_losses = 0
        value_losses = 0
        
        for episode in episodes:
            policy_loss, value_loss = self.compute_loss(episode, gamma)
            policy_losses += policy_loss
            value_losses += value_loss
        
        avg_policy_loss = policy_losses / len(episodes)
        avg_value_loss = value_losses / len(episodes)
        
        optimizer.zero_grad()
        value_optimizer.zero_grad()
        
        avg_policy_loss.backward(retain_graph=True)
        optimizer.step()
        
        avg_value_loss.backward()
        value_optimizer.step()
    
    def train(self, num_iterations, batch_size, gamma, learning_rate):
        self.policy_net.train()
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

        avg_rewards = []
        
        for i in range(num_iterations):
            episodes = [self.run_episode() for _ in range(batch_size)]
            self.update_policy(episodes, optimizer, value_optimizer, gamma)
            
            if i % 10 == 0:
                avg_reward = self.evaluate(10)
                avg_rewards.append(avg_reward)
            
            if(num_iterations > 1000):
                if i % 1000 == 0:
                    print(f"Episode {i}, Average Reward: {np.mean(avg_rewards[-1000:]):.2f}")
            else:
                if i % 100 == 0:
                    print(f"Episode {i}, Average Reward: {np.mean(avg_rewards[-100:]):.2f}")
        
        return avg_rewards

class ActorCriticAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim, gamma=0.99, learning_rate=0.01, results_dir='results/actor_critic'):
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.env = env
        self.policy_net_a2c = PolicyNet(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNet(state_dim, hidden_dim)

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.a2c = ActorCriticPolicyGradient(env, self.policy_net_a2c, self.value_net)
    
    def select_action(self, state):
        return self.a2c.select_action(state)
    
    def train(self, num_episodes):
        self.rewards = self.a2c.train(num_iterations=num_episodes, batch_size=10, gamma=self.gamma, learning_rate=self.learning_rate)
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
        plt.title('Actor-Critic Performance')
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
        self.policy_net_a2c.eval()
        total_rewards = 0
        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
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
            name_prefix='best_actor_critic_play'
        )
        
        # Reset environment and play the best episode
        state, _ = record_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose action based on learned policy
            action = self.select_action(state)
            state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Best Play Recorded. Total Reward: {total_reward}")
        print(f"Video saved in {video_dir}")
        record_env.close()
