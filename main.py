import gymnasium as gym
from agents.QLearningAgent import QLearningAgent
import os

def main():    
    # Render mode is important for video recording
    env = gym.make('FrozenLake-v1', render_mode='rgb_array')
    
    agent = QLearningAgent(
        env, 
        learning_rate=0.8, 
        discount_factor=0.95, 
        epsilon=1.0, 
        epsilon_decay=0.99, 
        epsilon_min=0.01
    )
    
    rewards = agent.train(num_episodes=10000)
    agent.evaluate()
    agent.plot_rewards()
    agent.record_best_play()
    env.close()

if __name__ == "__main__":
    main()