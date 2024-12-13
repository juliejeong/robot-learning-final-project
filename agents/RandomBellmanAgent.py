import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class RandomBellmanAgent:
    def __init__(self, env, discount_factor=0.95, results_dir='results/random_bellman_agent'):
        """
        Initialize the Random Bellman Agent.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.episode_rewards = []
        self.state_values = np.zeros(self.env.observation_space.n)  # Value function initialization

    def choose_action(self, state):
        """
        Choose a random action from the action space.
        """
        return self.env.action_space.sample()

    def bellman_update(self, state, reward, next_state, done):
        """
        Perform a Bellman update for state value approximation.
        """
        if not done:
            self.state_values[state] += 0.1 * (reward + self.discount_factor * self.state_values[next_state] - self.state_values[state])
        else:
            self.state_values[state] += 0.1 * (reward - self.state_values[state])

    def train(self, num_episodes=10000):
        """
        Train the agent using random actions and Bellman updates.
        """
        avg_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Perform Bellman update
                self.bellman_update(state, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)

            if (episode % 10 == 0):
                avg_rewards.append(np.mean(self.episode_rewards[-10:]))
                # print(np.mean(self.episode_rewards[-10:]))

            # Log progress every 1000 episodes
            if episode % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-1000:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        # return self.episode_rewards
        return avg_rewards

    def evaluate(self, num_eval_episodes=100):
        """
        Evaluate the agent over multiple episodes, using random actions.
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

         # Save evaluation results
        data_dir = os.path.join(self.results_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        with open(os.path.join(data_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Average Reward: {avg_reward}\n")
            f.write(f"Number of Evaluation Episodes: {num_eval_episodes}")

        return avg_reward

    def plot_rewards(self):
        """
        Plot and save the cumulative average rewards over episodes.
        """
        plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1))
        plt.title('RandomBellmanAgent Training Performance')
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
            name_prefix='random_bellman_agent_play'
        )

        state, _ = record_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            state, reward, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Random Bellman agent play recorded. Total Reward: {total_reward}")
        print(f"Video saved to {video_dir}")
        record_env.close()