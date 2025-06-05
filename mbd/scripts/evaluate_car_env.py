import jax
import jax.numpy as jnp
from mbd.envs.car_env import CarEnv
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np # For np.mean in plotting
import onnxruntime as ort

# Placeholder for policy import
# from your_policy_module import your_policy_function

def evaluate_policy(env, policy, num_episodes=2, history_len=10, max_episode_steps=100):
    all_episodes_data = []
    jit_step = jax.jit(env.step)

    for episode_idx in range(num_episodes):
        state = env.reset(jax.random.PRNGKey(episode_idx+100))
        episode_reward_sum = 0.0
        
        obs_history = collections.deque(maxlen=history_len)
        current_obs_for_hist = state.obs
        for _ in range(history_len):
            obs_history.append(jnp.zeros_like(current_obs_for_hist))
        obs_history.append(current_obs_for_hist)

        episode_states_history = [] # To store pipeline_state.q for plotting
        episode_rewards_history = [] # To store rewards for plotting

        for step_idx in range(max_episode_steps):
            # Store current state for plotting (q has [x,y,theta])
            episode_states_history.append(state.pipeline_state.q)

            policy_obs_input = jnp.array(list(obs_history))
            
            # print(f"Policy obs input shape: {policy_obs_input.shape}") # User's debug line
            action = policy(policy_obs_input)
            
            state = jit_step(state, action)
            
            obs_history.append(state.obs)
            episode_rewards_history.append(state.reward)
            episode_reward_sum += state.reward
            
            if jnp.isnan(state.obs).any() or jnp.isnan(state.reward):
                print(f"[DEBUG] NaN detected in episode {episode_idx} at step {step_idx}")
                # break # Optionally break if NaN detected
            
            # print(f"Episode {episode_idx}, Step {step_idx} reward: {state.reward:.4f}")
            if state.done:
                break
        
        final_velocity = state.pipeline_state.qd[:3]
        all_episodes_data.append({
            "states": jnp.array(episode_states_history),
            "rewards": jnp.array(episode_rewards_history)
        })
        print(f"Episode {episode_idx} total reward: {episode_reward_sum:.4f}")
        print(f"Episode {episode_idx} final velocity (vx, vy, wz): {final_velocity}")
        print(f"Episode {episode_idx} final position (x, y, theta): {state.pipeline_state.q}")
    
    avg_reward = np.mean([jnp.sum(ep_data["rewards"]) for ep_data in all_episodes_data])
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.4f}")
    return all_episodes_data

def plot_evaluation_trajectories(episodes_data, env, save_path="../results/plots/evaluation_trajectories.png"):
    """Plot all trajectories and rewards from evaluation."""
    num_episodes = len(episodes_data)
    selected_indices = np.arange(num_episodes) # Plot all trajectories

    plt.figure(figsize=(15, 10))
    ax1 = None

    # Get obstacle data from env
    obstacles_pos = np.array(env.obstacle_positions_xy)
    obstacle_rad = env.obstacle_radius

    for i, episode_idx in enumerate(selected_indices):
        episode = episodes_data[episode_idx]
        traj_states = np.array(episode["states"]) # q has x, y, theta
        traj_rewards = np.array(episode["rewards"])

        # Plot position (first two dimensions are x,y position, third is theta)
        if ax1 is None:
            ax1 = plt.subplot(2, 1, 1)
            # Plot obstacles ONCE
            if obstacles_pos.shape[0] > 0:
                print(f"Plotting {obstacles_pos.shape[0]} obstacles from environment.")
                for obs_idx in range(obstacles_pos.shape[0]):
                    circle = patches.Circle((obstacles_pos[obs_idx, 0], obstacles_pos[obs_idx, 1]), 
                                            obstacle_rad, 
                                            color='gray', 
                                            alpha=0.5, 
                                            fill=True,
                                            zorder=0)
                    ax1.add_patch(circle)
        else:
            plt.subplot(2, 1, 1) # Ensure this subplot is active
        
        plt.plot(traj_states[:, 0], traj_states[:, 1], 
                 label=f'Traj {episode_idx} (R={np.sum(traj_rewards):.2f})', zorder=1)
        
        arrow_spacing = max(1, len(traj_states) // 20) # Show about 20 arrows
        arrow_indices = np.arange(0, len(traj_states), arrow_spacing)
        x_arrows = traj_states[arrow_indices, 0]
        y_arrows = traj_states[arrow_indices, 1]
        theta_arrows = traj_states[arrow_indices, 2]
        arrow_length = 0.08
        dx = arrow_length * np.cos(theta_arrows)
        dy = arrow_length * np.sin(theta_arrows)

        plt.quiver(
            x_arrows, y_arrows, dx, dy,
            color='k', alpha=0.8,
            angles='xy', scale_units='xy', scale=1.0,
            width=0.008, headwidth=2, headlength=3, headaxislength=2,
            zorder=2
        )
        
    if ax1: # Only configure if plyition')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title('Evaluated Trajectory Positions with Orientation and Obstacles')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True)
        ax1.legend(fontsize='small')
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.5, 0.9)

    # Plot rewards
    
    ax2 = plt.subplot(2, 1, 2)
    for i, episode_idx in enumerate(selected_indices):
        episode = episodes_data[episode_idx]
        traj_rewards = np.array(episode["rewards"])
        plt.plot(traj_rewards, label=f'Traj {episode_idx}')
    
    ax2.set_title('Rewards Over Time for All Trajectories')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    ax2.legend(fontsize='small')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Plotted {len(selected_indices)} evaluated trajectories.")
    print(f"Plot saved to {save_path}")

def create_onnx_policy(model_path, history_len, obs_dim, action_size):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # Assuming the model might output more than just action, or action might not be the first output
    # output_names = [output.name for output in session.get_outputs()]

    def policy_fn(obs_sequence: np.ndarray) -> np.ndarray:
        # obs_sequence is JAX array from deque, shape (history_len, obs_dim)
        # Convert to numpy, float32, and add batch dimension
        obs_np = np.array(obs_sequence, dtype=np.float32)
        if obs_np.shape != (history_len, obs_dim):
             raise ValueError(f"Expected obs_sequence shape {(history_len, obs_dim)}, got {obs_np.shape}")
        
        batched_obs = np.expand_dims(obs_np, axis=0) # Shape: (1, history_len, obs_dim)
        
        # Run inference
        # Assuming the primary output (index 0) is the action
        action_output = session.run(None, {input_name: batched_obs})[0]
        
        # print(f"Action output shape: {action_output.shape}")
        return action_output[0][0]

    return policy_fn

def main():
    env = CarEnv()
    history_len = 10 # Should match what the ONNX model was trained with
    obs_dim = env.observation_size
    action_size = env.action_size

    onnx_model_path = "./results/diffusion.onnx"
    
    # Create the ONNX policy function
    try:
        onnx_policy = create_onnx_policy(onnx_model_path, history_len, obs_dim, action_size)
        print(f"Successfully loaded ONNX policy from {onnx_model_path}")
    except Exception as e:
        print(f"Error loading ONNX policy: {e}")
        print("Falling back to dummy policy.")
        def dummy_policy(obs_sequence):
            return jnp.zeros(env.action_size)
        onnx_policy = dummy_policy # Fallback to dummy if ONNX fails

    num_eval_episodes = 5
    max_steps = 500
    evaluated_data = evaluate_policy(env, onnx_policy, num_episodes=num_eval_episodes, 
                                     history_len=history_len, max_episode_steps=max_steps)
    plot_evaluation_trajectories(evaluated_data, env)

if __name__ == "__main__":
    main() 