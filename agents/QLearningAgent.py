import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

class QLearningAgent:
    def __init__(self, 
                 env, 
                 learning_rate=0.8, 
                 discount_factor=0.95, 
                 epsilon=1.0, 
                 epsilon_decay=0.99, 
                 epsilon_min=0.01,
                 results_dir='results/q_learning'):
        """
        Initialize Q-Learning Agent with organized results directory
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
        
        # Initialize Q-table
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        
        # Tracking rewards
        self.episode_rewards = []
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def train(self, num_episodes=10000):
        """
        Train the agent and return episode rewards
        """
        avg_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update Q-table
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # if (episode % 10 == 0):
            #     self.episode_rewards.append(total_reward)

            self.episode_rewards.append(total_reward)

            if (episode % 10 == 0):
                avg_rewards.append(np.mean(self.episode_rewards[-10:]))

            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(self.episode_rewards[-1000:]):.2f}")
        
        return avg_rewards
        # return np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1)
        # return self.episode_rewards
    
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
            name_prefix='best_frozen_lake_play'
        )
        
        # Find the episode with the highest reward
        best_episode_reward = max(self.episode_rewards)
        
        # Reset environment and play the best episode
        state, _ = record_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose best action based on learned Q-table
            action = np.argmax(self.q_table[state])
            
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
        plt.title('Q-Learning Performance on Frozen Lake')
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
                action = np.argmax(self.q_table[state])
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