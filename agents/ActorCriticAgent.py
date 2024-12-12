import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCriticAgent:
    def __init__(self, 
                 env, 
                 learning_rate=0.01, 
                 gamma=0.99, 
                 results_dir='results/actor_critic'):
        """
        Initialize Actor-Critic Agent
        
        Args:
            env (gym.Env): Gymnasium environment
            learning_rate (float): Learning rate for optimization
            gamma (float): Discount factor for future rewards
            results_dir (str): Directory to save results
        """
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Determine input and output dimensions
        self.input_dim = env.observation_space.n
        self.output_dim = env.action_space.n
        
        # Actor-Critic Network
        self.network = ActorCriticNetwork(self.input_dim, self.output_dim)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Tracking rewards
        self.episode_rewards = []
    
    def choose_action(self, state):
        """
        Choose action using the actor's policy
        
        Args:
            state (int): Current environment state
        
        Returns:
            int: Selected action
        """
        # Convert state to one-hot encoding
        state_tensor = self._state_to_tensor(state)
        
        # Get action probabilities
        action_probs = self.network.actor(state_tensor)
        
        # Sample action based on probabilities
        action = torch.multinomial(action_probs, 1).item()
        
        return action
    
    def _state_to_tensor(self, state):
        """
        Convert state to one-hot tensor
        
        Args:
            state (int): Environment state
        
        Returns:
            torch.Tensor: One-hot encoded state
        """
        state_tensor = torch.zeros(self.input_dim)
        state_tensor[state] = 1.0
        return state_tensor
    
    def train(self, num_episodes=10000):
        """
        Train the Actor-Critic agent
        
        Args:
            num_episodes (int): Number of training episodes
        
        Returns:
            list: Episode rewards
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Convert states to tensors
                state_tensor = self._state_to_tensor(state)
                next_state_tensor = self._state_to_tensor(next_state)
                
                # Compute TD error
                current_value = self.network.critic(state_tensor)
                next_value = self.network.critic(next_state_tensor)
                
                # Compute target
                target = reward + self.gamma * next_value * (not done)
                td_error = target - current_value
                
                # Compute loss
                critic_loss = td_error.pow(2)
                
                # Actor loss using policy gradient
                log_prob = torch.log(self.network.actor(state_tensor)[action])
                actor_loss = -log_prob * td_error.detach()
                
                # Combined loss
                loss = critic_loss + actor_loss
                
                # Backpropagate and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state = next_state
                total_reward += reward
            
            self.episode_rewards.append(total_reward)
            
            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(self.episode_rewards[-1000:]):.2f}")
        
        return self.episode_rewards
    
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
            episode_trigger=lambda episode: True,
            name_prefix='best_frozen_lake_play'
        )
        
        # Reset environment and play the best episode
        state, _ = record_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose best action based on actor's policy
            action = torch.argmax(self.network.actor(self._state_to_tensor(state))).item()
            
            # Take action
            state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            
            total_reward += reward
        
        print(f"Best Play Recorded. Total Reward: {total_reward}")
        print(f"Video saved in {video_dir}")
        
        # Close the recording environment
        record_env.close()
    
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
        plt.title('Actor-Critic Performance on Frozen Lake')
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
        
        Args:
            num_eval_episodes (int): Number of episodes to evaluate
        
        Returns:
            float: Average reward during evaluation
        """
        total_rewards = 0
        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = torch.argmax(self.network.actor(self._state_to_tensor(state))).item()
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

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Actor-Critic Neural Network
        
        Args:
            input_dim (int): Input dimension (state space size)
            output_dim (int): Output dimension (action space size)
        """
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state
        
        Returns:
            tuple: Actor output (action probabilities), Critic output (state value)
        """
        return self.actor(x), self.critic(x)