import jax
import os
from jax import numpy as jnp
import time
from tqdm import tqdm
import tyro
import zarr
import numpy as np
from jax import tree_util
import sys
import os

# Add the parent directory to sys.path to allow direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mbd.planners.mbd_planner_new import Args, MBDPI, rollout_us
import mbd.envs

def save_trajectory_data(args: Args, rewss, pipeline_statess, us, obs, state_init, env, zroot=None, episode_num=None):
    """Save trajectory data to Zarr format, appending to existing data if provided."""
    if not args.save_data:
        return
    
    print(f"Saving trajectory data for episode {episode_num}...")
    
    # Create output directory
    data_output_dir = f"../results/{args.env_name}"
    os.makedirs(data_output_dir, exist_ok=True)
    
    # Create or open Zarr file
    zarr_file_path = os.path.join(data_output_dir, "mbd_trajectories.zarr")
    
    try:
        if zroot is None:
            # Create new Zarr file
            zroot = zarr.open_group(zarr_file_path, "w")
            zdata = zroot.create_group("data")
            zmeta = zroot.create_group("meta")
            
            # Initialize datasets
            obs_dim = env.observation_size
            action_dim = env.action_size
            
            # Create resizable datasets
            zdata.create_dataset("state", shape=(0, obs_dim), dtype='float32', chunks=(1000, obs_dim))
            zdata.create_dataset("action", shape=(0, action_dim), dtype='float32', chunks=(1000, action_dim))
            zdata.create_dataset("reward", shape=(0,), dtype='float32', chunks=(1000,))
            zmeta.create_dataset("episode_ends", shape=(0,), dtype='int64', chunks=(100,))
            
            # Initialize metadata
            zmeta.attrs['num_trajectories'] = 0
            zmeta.attrs['horizon'] = args.Hsample
            zmeta.attrs['obs_dim'] = obs_dim
            zmeta.attrs['action_dim'] = action_dim
            zmeta.attrs['env_name'] = args.env_name
            zmeta.attrs['total_steps'] = 0
        else:
            zdata = zroot['data']
            zmeta = zroot['meta']
        
        # Select best trajectories based on mean reward
        trajectory_rewards = rewss.mean(axis=1)  # Mean reward per trajectory
        best_indices = jnp.argsort(trajectory_rewards)[-args.max_trajectories_to_save:]  # Top trajectories
        
        # print(f"Selected {len(best_indices)} best trajectories out of {len(trajectory_rewards)} total")
        # print(f"Best trajectory rewards: {trajectory_rewards[best_indices]}")
        
        # Extract best trajectories
        best_rewss = rewss[best_indices]  # Shape: (num_best, Hsample)
        best_us = us[best_indices]  # Shape: (num_best, Hsample, action_dim)
        best_obs = obs[best_indices]
        
        # Flatten data for storage
        num_best_trajectories = best_obs.shape[0]
        horizon = best_obs.shape[1]
        total_steps = num_best_trajectories * horizon
        
        flat_obs = best_obs.reshape(total_steps, env.observation_size)
        flat_actions = best_us.reshape(total_steps, env.action_size)
        flat_rewards = best_rewss.reshape(total_steps)
        
        # Convert to numpy for Zarr storage
        flat_obs_np = np.array(flat_obs, dtype=np.float32)
        flat_actions_np = np.array(flat_actions, dtype=np.float32)
        flat_rewards_np = np.array(flat_rewards, dtype=np.float32)
        
        # Get current sizes
        current_size = zdata['state'].shape[0]
        new_size = current_size + total_steps
        
        # Resize datasets - Zarr resize takes the new shape directly
        zdata['state'].resize((new_size, env.observation_size))
        zdata['action'].resize((new_size, env.action_size))
        zdata['reward'].resize((new_size,))
        
        # Append new data
        zdata['state'][current_size:new_size] = flat_obs_np
        zdata['action'][current_size:new_size] = flat_actions_np
        zdata['reward'][current_size:new_size] = flat_rewards_np
        
        # Update episode ends
        current_episodes = zmeta['episode_ends'].shape[0]
        new_episode_ends = np.arange(horizon, total_steps + 1, horizon, dtype=np.int64) + current_size
        zmeta['episode_ends'].resize((current_episodes + len(new_episode_ends),))
        zmeta['episode_ends'][current_episodes:] = new_episode_ends
        
        # Update metadata
        zmeta.attrs['num_trajectories'] += num_best_trajectories
        zmeta.attrs['total_steps'] = new_size
        
        # print(f"Successfully appended {total_steps} steps from {num_best_trajectories} trajectories")
        print(f"Total steps in dataset: {new_size}")
        print(f"Total trajectories in dataset: {zmeta.attrs['num_trajectories']}")
        
        return zroot
        
    except Exception as e:
        print(f"Error saving trajectory data: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_single_episode(args: Args, env, mbdpi, rng):
    """Run a single episode of the MBD planner."""
    # Reset environment
    rng, rng_reset = jax.random.split(rng)
    state_init = env.reset(rng_reset)
    
    # Initialize control nodes
    YN = jnp.zeros([args.Hnode + 1, mbdpi.nu])
    
    # Run diffusion process
    rng_exp, rng = jax.random.split(rng)
    Y0 = YN
    
    # Run diffusion steps
    Ybars = []
    with tqdm(range(args.Ndiffuse), desc="Diffusing") as pbar:
        for i in pbar:
            rng, Y0, info = mbdpi.reverse_once(state_init, rng, Y0, mbdpi.sigma_control* (args.traj_diffuse_factor ** i))
            Ybars.append(Y0)
            pbar.set_postfix({"rew": f"{info['rews'].mean():.2e}"})
    
    # Generate final trajectories
    rng, Y0, info = mbdpi.reverse_once(state_init, rng, Y0, mbdpi.sigma_control)
    
    # Run rollout to get trajectories
    rewss = info['rews']
    pipeline_statess = info['pipeline_statess']
    us = info['us']
    obs = info['obs']
    # import pdb; pdb.set_trace()
    return rewss, pipeline_statess, us, obs, state_init, rng

def main():
    # Parse arguments
    args = tyro.cli(Args)
    
    # Create output directory
    os.makedirs(args.data_output_dir, exist_ok=True)
    
    # Initialize environment and planner
    env = mbd.envs.get_env(args.env_name)
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    mbdpi = MBDPI(args, env)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(args.seed)
    
    # Number of episodes to run
    num_episodes = 150  # You can adjust this number
    
    print(f"Starting {num_episodes} episodes of data collection...")
    
    # Initialize Zarr file
    zroot = None
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes}")
        
        # Run single episode
        rewss, pipeline_statess, us, obs, state_init, rng = run_single_episode(args, env, mbdpi, rng)
        
        # Save trajectory data
        zroot = save_trajectory_data(args, rewss, pipeline_statess, us, obs, state_init, env, zroot, episode + 1)
        
        
        print(f"Episode {episode + 1} completed. Data saved.")
        
        
    
    print("\nData collection completed!")

if __name__ == "__main__":
    main() 