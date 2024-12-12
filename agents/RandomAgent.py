import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class RandomAgent:
    def __init__(self, env, results_dir='results/random_agent'):
        """
        Initialize the RandomAgent with necessary directories for results.
        """
        self.env = env
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.episode_rewards = []

    def choose_action(self, state):
        """
        Choose a random action from the action space.
        """
        return self.env.action_space.sample()

    def train(self, num_episodes=100):
        """
        Train the random agent by running it for a specified number of episodes.
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            self.episode_rewards.append(total_reward)

            # Log progress
            if episode % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        return self.episode_rewards

    def evaluate(self, num_eval_episodes=100):
        """
        Evaluate the random agent by calculating average rewards over multiple episodes.
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
        print(f"Evaluation Complete: Average Reward = {avg_reward:.2f}")

        return avg_reward

    def plot_rewards(self):
        """
        Plot and save the cumulative average rewards for the training episodes.
        """
        plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1))
        plt.title('RandomAgent Training Performance')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Average Reward')
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, 'rewards_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Rewards plot saved to {plot_path}")
        
    def record_best_play(self):
        """
        Record a random gameplay episode as a video.
        """
        video_dir = os.path.join(self.results_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)

        record_env = gym.wrappers.RecordVideo(
            self.env,
            video_dir,
            episode_trigger=lambda _: True,
            name_prefix='random_agent_play'
        )

        state, _ = record_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Random agent play recorded. Total Reward: {total_reward}")
        print(f"Video saved to {video_dir}")
        
        record_env.close()