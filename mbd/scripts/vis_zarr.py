import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.patches as patches # Keep for drawing circles
from mbd.utils.trajectory_manager import TrajectoryManager # Import the new class

def plot_random_trajectories(zarr_path, num_trajectories_to_plot=50):
    """Plot random trajectories from the Zarr file using TrajectoryManager."""
    try:
        manager = TrajectoryManager(zarr_path)
        print(f"Successfully loaded trajectories using TrajectoryManager from: {zarr_path}")
    except (IOError, ValueError) as e:
        print(f"Error initializing TrajectoryManager: {e}")
        return

    total_trajectories_available = manager.num_trajectories
    if total_trajectories_available == 0:
        print("No trajectories found to plot.")
        return

    # Hardcoded obstacle data (remains the same as it's specific to this visualization script)
    obstacles = [
        {'pos': (0.7, 0.7), 'radius': 0.45, 'name': 'obstacle1'},
        {'pos': (-0.7, 0.7), 'radius': 0.50, 'name': 'obstacle2'},
        {'pos': (0.7, -0.7), 'radius': 0.50, 'name': 'obstacle3'},
        {'pos': (-0.7, -0.7), 'radius': 0.55, 'name': 'obstacle4'},
        {'pos': (0.7, 0.0), 'radius': 0.55, 'name': 'obstacle5'},
        {'pos': (-0.7, 0.0), 'radius': 0.55, 'name': 'obstacle6'}
    ]
    
    # Randomly select trajectories
    num_to_plot = min(num_trajectories_to_plot, total_trajectories_available)
    selected_indices = random.sample(range(total_trajectories_available), 1000)
    traj_rewards_list = list()
    for idx in selected_indices:
        _, _, traj_rewards = manager.get_trajectory(idx)
        traj_rewards_list.append(np.sum(traj_rewards))
    # sort the traj_rewards_list by the sum of the rewards
    top_k_indices = np.argsort(traj_rewards_list)[-num_trajectories_to_plot:][::-1]
    print(top_k_indices)    
    
    plt.figure(figsize=(15, 10))
    ax1 = None 

    for idx in top_k_indices:
        traj_states, _, traj_rewards = manager.get_trajectory(idx)
        
        if traj_states is None or traj_rewards is None:
            print(f"Warning: Could not retrieve trajectory {idx}. Skipping.")
            continue
        if len(traj_states) == 0:
            print(f"Warning: Trajectory {idx} is empty. Skipping.")
            continue

        # Plot position (assuming first two dimensions are x,y position)
        if ax1 is None:
            ax1 = plt.subplot(2, 1, 1)
            if obstacles:
                # print(f"Plotting {len(obstacles)} hardcoded obstacles.") # Less verbose
                for obs_data in obstacles:
                    circle = patches.Circle((obs_data['pos'][0], obs_data['pos'][1]), 
                                            obs_data['radius'], 
                                            color='gray', alpha=0.5, fill=True, zorder=0)
                    ax1.add_patch(circle)
        else:
            plt.subplot(2, 1, 1) 
        
        plt.plot(traj_states[:, 0], traj_states[:, 1], 
                 label=f'Traj {idx} (R={np.mean(traj_rewards):.2f})', zorder=1)
        
        arrow_spacing = max(1, len(traj_states) // 20) # Show about 20 arrows
        arrow_indices = np.arange(0, len(traj_states), arrow_spacing)
        x_arrows = traj_states[arrow_indices, 0]
        y_arrows = traj_states[arrow_indices, 1]
        theta_arrows = traj_states[arrow_indices, 2]
        arrow_length = 0.08
        dx = arrow_length * np.cos(theta_arrows)
        dy = arrow_length * np.sin(theta_arrows)

        plt.quiver(x_arrows, y_arrows, dx, dy, color='k', alpha=0.8,
                   angles='xy', scale_units='xy', scale=1.0,
                   width=0.008, headwidth=2, headlength=3, headaxislength=2, zorder=2)
        
    if ax1: # Only configure if plots were made
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title('Trajectory Positions with Orientation and Obstacles')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True)
        ax1.legend(fontsize='small') # Added legend display
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.5, 0.9)

    # Plot rewards for selected trajectories
    ax2 = plt.subplot(2, 1, 2)
    plotted_rewards = False
    for idx in top_k_indices:
        _, _, traj_rewards = manager.get_trajectory(idx)
        if traj_rewards is not None and len(traj_rewards) > 0:
            plt.plot(traj_rewards, label=f'Traj {idx}')
            plotted_rewards = True
    
    if plotted_rewards:
        ax2.set_title('Rewards Over Time for Selected Trajectories')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        ax2.legend(fontsize='small') # Added legend display
    else:
        ax2.text(0.5, 0.5, "No reward data to plot for selected trajectories", ha='center', va='center')

    plt.tight_layout()
    
    save_dir = "../results/plots"
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(save_dir, "random_trajectories_managed.png")
    plt.savefig(save_file_path)
    plt.close()
    
    print(f"Plotted {len(selected_indices)} random trajectories.")
    print(f"Plot saved to {save_file_path}")

if __name__ == "__main__":
    default_zarr_path = "../results/car_env/mbd_trajectories.zarr"
    plot_random_trajectories(default_zarr_path, num_trajectories_to_plot=5) # User change, from 20 to 5
