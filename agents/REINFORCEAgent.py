import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from functools import reduce
import os

class Policy_network(tf.keras.Model):
  def __init__(self, state_dim, action_dim, hidden_units=(64, 64), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
    super(Policy_network, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), dtype=tf.float32, name='input')
    self.hidden_layers = []
    for i in range(len(hidden_units)):
      self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, name='dense{}'.format(i)))

    self.output_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer, name='output')

    self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32))

  @tf.function
  def call(self, input, activation='softmax'):
    z = self.input_layer(input)
    for layer in self.hidden_layers:
      z = layer(z)
    output = self.output_layer(z)
    output = tf.keras.activations.get(activation)(output)
    return output

class REINFORCEAgent:
  def __init__(self, env, gamma=0.9, alpha=0.001, max_episode=1000):
    self.env = env
    self.state_dim = env.observation_space.n
    self.action_dim = env.action_space.n
    
    self.nrow = env.nrow
    self.ncol = env.ncol 
    
    self.gamma = gamma
    self.alpha = alpha
    
    self.max_episode = max_episode

    self.policy = Policy_network(self.state_dim, self.action_dim)
    self.optimizer = tf.keras.optimizers.Adam(self.alpha)

  def choose_action(self, s):
    s = np.eye(self.state_dim)[s]  # one-hot encoding
    s = np.expand_dims(s, axis=0)  # expand state dimension

    policy = self.policy(s)
    dist = tfp.distributions.Categorical(probs=policy)
    action = dist.sample().numpy()

    return action[0]

  def run(self):
    self.success = 0

    for episode in range(self.max_episode):
      observation, _ = self.env.reset()

      done = False
      episode_reward = 0
      local_step = 0

      trajectory = []
      policy_prob = self.policy(np.eye(self.state_dim)).numpy()

      while not done:
        action = self.choose_action(observation)
        next_observation, reward, done, _, _ = self.env.step(action)

        # give penalty for staying in ground
        if reward == 0:
          reward = -0.001

        # give penalty for falling into the hole
        if done and next_observation != 15:
          reward = -1

        if local_step == 100:
          done = True  # prevent infinite episode
          reward = -1

        if observation == next_observation:  # prevent meaningless actions
          reward = -1

        trajectory.append({"s": observation, "a": action, "r": reward})

        observation = next_observation

        episode_reward += reward
        local_step += 1

      if observation == 15:
        self.success += 1

      reward_list = np.array(list(map(lambda x: x['r'], trajectory)))

      # stochastic gradient descent
      for i in range(len(trajectory)):
        with tf.GradientTape() as tape:
          G = reduce(lambda acc, x: acc + (self.gamma ** x[0]) * x[1], enumerate(reward_list[i:]), 0)

          s = np.eye(self.state_dim)[trajectory[i]['s']]  # one-hot encoding
          s = np.expand_dims(s, axis=0)  # expand state dimension

          policy = self.policy(s)
          dist = tfp.distributions.Categorical(probs=policy)
          log_policy = tf.reshape(dist.log_prob(trajectory[i]['a']), (-1, 1))

          loss = -(self.gamma ** i) * G * log_policy

        variables = self.policy.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))
        def plot_rewards(self):
          """
          Plot cumulative average rewards and save to results directory
          """
          import matplotlib.pyplot as plt

          # Create plots subdirectory
          plots_dir = os.path.join(self.results_dir, 'plots')
          os.makedirs(plots_dir, exist_ok=True)
          
          # Create the plot
          plt.figure(figsize=(10, 5))
          plt.plot(np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1))
          plt.title('REINFORCE Agent Performance on Frozen Lake')
          plt.xlabel('Episodes')
          plt.ylabel('Average Reward')
          plt.tight_deactivatelayout()
          
          # Save the plot
          plot_path = os.path.join(plots_dir, 'rewards_plot.png')
          plt.savefig(plot_path)
          plt.close()
          
          print(f"Rewards plot saved to {plot_path}")
        # Create plots subdirectory
        plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1))
        plt.title('REINFORCE Agent Performance on Frozen Lake')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, 'rewards_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Rewards plot saved to {plot_path}")

  def evaluate(self, num_episodes=100):
    total_rewards = []
    for _ in range(num_episodes):
      observation, _ = self.env.reset()
      done = False
      episode_reward = 0
      while not done:
        action = self.choose_action(observation)
        observation, reward, done, _, _ = self.env.step(action)
        episode_reward += reward
      total_rewards.append(episode_reward)
    return np.mean(total_rewards)

  def record_best_play(self):
    observation, _ = self.env.reset()
    done = False
    trajectory = []
    while not done:
      action = self.choose_action(observation)
      next_observation, reward, done, _, _ = self.env.step(action)
      trajectory.append((observation, action, reward))
      observation = next_observation
    return trajectory
