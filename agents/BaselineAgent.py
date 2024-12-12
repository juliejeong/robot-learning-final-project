import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

class BaselineAgent:
    def __init__(self, 
                 env, 
                 learning_rate=0.8, 
                 discount_factor=0.95, 
                 epsilon=1.0, 
                 epsilon_decay=0.99, 
                 epsilon_min=0.01,
                 prob = 0.5,
                 results_dir='results/baseline_agent'):
        """
        Initialize Baseline Agent with organized results directory
        """
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Tracking rewards
        self.episode_rewards = []
    
    def choose_action(self, state):
        """
        Choose action randomly
        """
        return np.random.choice([0, 1, 2, 3])
    
    def train(self, num_episodes=10000):
      """
      Train the agent and return episode rewards
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
          
          state = next_state
          total_reward += reward
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards.append(total_reward)
        
        # Print progress
        if episode % 1000 == 0:
          print(f"Episode {episode}, Average Reward: {np.mean(self.episode_rewards[-1000:]):.2f}")
      
      return self.episode_rewards
    
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
      plt.title('Random Action Performance on Frozen Lake')
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
      total_rewards = 0
      for _ in range(num_eval_episodes):
        state, _ = self.env.reset()
        done = False
        while not done:
          action = self.choose_action(state)
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
