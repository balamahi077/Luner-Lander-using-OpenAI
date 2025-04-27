import os
import torch
import numpy as np
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

env_name = 'LunarLander-v3'
env = gym.make(env_name, render_mode="human")  

# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Wrap the environment with Monitor
env = Monitor(env, log_dir)

# Define the neural network architecture - larger network for LunarLander
nn_layers = [256, 256]  # Increased network size for better performance
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=nn_layers
)

# Create and train the model with optimized parameters for LunarLander
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0005,  # Reduced learning rate for more stable learning
    batch_size=128,        # Increased batch size
    buffer_size=100000,    # Larger buffer for better experience replay
    learning_starts=5000,  # More initial random steps
    gamma=0.99,
    tau=0.005,            # Soft update parameter
    target_update_interval=1000,
    train_freq=(1, "step"),
    max_grad_norm=10,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.2,  # Longer exploration period
    gradient_steps=1,
    seed=1,
    verbose=1
)

# Create evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Train the model for longer
print("Starting training...")
model.learn(
    total_timesteps=50000,  # Increased training time
    callback=eval_callback,
    progress_bar=True
)

# Save the model
model.save("lunar_lander_dqn")

# Test the trained model
print("\nTesting the trained model...")
obs = env.reset()[0]
done = False
total_reward = 0
episode_count = 0

# Test for multiple episodes
for episode in range(5):  # Test 5 episodes
    obs = env.reset()[0]
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info, _ = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1} - Total reward: {total_reward}")

env.close()

