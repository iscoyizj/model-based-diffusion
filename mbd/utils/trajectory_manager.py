import zarr
import numpy as np

class TrajectoryManager:
    """Manages trajectories loaded from a Zarr file."""

    def __init__(self, zarr_path):
        """
        Initializes the TrajectoryManager by loading data from the Zarr file.

        Args:
            zarr_path (str): Path to the Zarr file.
        """
        try:
            zroot = zarr.open_group(zarr_path, "r")
            zdata = zroot['data']
            zmeta = zroot['meta']

            self.states = zdata['state'][:]
            self.actions = zdata['action'][:]
            self.rewards = zdata['reward'][:]
            self.episode_ends = zmeta['episode_ends'][:]
            
            self.horizon = zmeta.attrs.get('horizon', -1) # Use .get for robustness
            self.num_trajectories = zmeta.attrs.get('num_trajectories', len(self.episode_ends))

            if self.num_trajectories != len(self.episode_ends):
                print(f"Warning: num_trajectories in metadata ({self.num_trajectories}) "
                      f"does not match length of episode_ends ({len(self.episode_ends)}). "
                      f"Using length of episode_ends.")
                self.num_trajectories = len(self.episode_ends)
            
            self._validate_data()

        except Exception as e:
            raise IOError(f"Error loading or parsing Zarr file at {zarr_path}: {e}")

    def _validate_data(self):
        """Basic validation of loaded data."""
        if not (len(self.states) == len(self.actions) == len(self.rewards)):
            raise ValueError("Mismatch in lengths of states, actions, or rewards arrays.")
        if len(self.episode_ends) == 0 and self.num_trajectories > 0:
            raise ValueError("episode_ends is empty but num_trajectories > 0.")
        if len(self.episode_ends) > 0 and self.episode_ends[-1] > len(self.states):
            raise ValueError("Last episode_end index out of bounds for states array.")

    def get_trajectory(self, trajectory_index):
        """
        Retrieves a specific trajectory (states, actions, rewards).

        Args:
            trajectory_index (int): Index of the trajectory to retrieve.

        Returns:
            tuple: (states_traj, actions_traj, rewards_traj)
                   Returns (None, None, None) if index is out of bounds.
        """
        if not (0 <= trajectory_index < self.num_trajectories):
            print(f"Error: Trajectory index {trajectory_index} out of bounds (0-{self.num_trajectories-1}).")
            return None, None, None

        start_idx = 0 if trajectory_index == 0 else self.episode_ends[trajectory_index - 1]
        end_idx = self.episode_ends[trajectory_index]

        states_traj = self.states[start_idx:end_idx]
        actions_traj = self.actions[start_idx:end_idx]
        rewards_traj = self.rewards[start_idx:end_idx]
        
        return states_traj, actions_traj, rewards_traj

    def get_trajectory_reward_stats(self, trajectory_index):
        """
        Calculates reward statistics for a single trajectory.

        Args:
            trajectory_index (int): Index of the trajectory.

        Returns:
            dict: Contains 'sum', 'mean', 'min', 'max' reward for the trajectory.
                  Returns None if index is out of bounds or trajectory is empty.
        """
        _, _, rewards_traj = self.get_trajectory(trajectory_index)

        if rewards_traj is None or len(rewards_traj) == 0:
            return None
        
        return {
            'sum': np.sum(rewards_traj),
            'mean': np.mean(rewards_traj),
            'min': np.min(rewards_traj),
            'max': np.max(rewards_traj),
            'len': len(rewards_traj)
        }

    def get_all_trajectory_reward_stats(self):
        """
        Calculates reward statistics for all trajectories.

        Returns:
            list: A list of dictionaries, each containing reward stats for a trajectory.
        """
        all_stats = []
        for i in range(self.num_trajectories):
            stats = self.get_trajectory_reward_stats(i)
            if stats:
                all_stats.append(stats)
        return all_stats

    def get_overall_reward_stats(self):
        """
        Calculates overall reward statistics across all trajectories.

        Returns:
            dict: Contains 'mean_sum_reward' (mean of trajectory sums) and 
                  'mean_mean_reward' (mean of trajectory means).
                  Returns None if no trajectories or no stats available.
        """
        all_traj_stats = self.get_all_trajectory_reward_stats()
        if not all_traj_stats:
            return None

        sum_rewards_all_trajs = [stats['sum'] for stats in all_traj_stats]
        mean_rewards_all_trajs = [stats['mean'] for stats in all_traj_stats]

        return {
            'mean_sum_reward': np.mean(sum_rewards_all_trajs) if sum_rewards_all_trajs else 0,
            'std_sum_reward': np.std(sum_rewards_all_trajs) if sum_rewards_all_trajs else 0,
            'min_sum_reward': np.min(sum_rewards_all_trajs) if sum_rewards_all_trajs else 0,
            'max_sum_reward': np.max(sum_rewards_all_trajs) if sum_rewards_all_trajs else 0,
            'mean_mean_reward': np.mean(mean_rewards_all_trajs) if mean_rewards_all_trajs else 0,
            'num_trajectories_processed': len(all_traj_stats)
        }

if __name__ == '__main__':
    # Example Usage (assuming you have a Zarr file at this path)
    # Replace with the actual path to your Zarr file
    example_zarr_path = "../results/car_env/mbd_trajectories.zarr" 
    
    print(f"Attempting to load trajectories from: {example_zarr_path}")
    try:
        manager = TrajectoryManager(example_zarr_path)
        print(f"Successfully loaded {manager.num_trajectories} trajectories.")
        print(f"Data horizon: {manager.horizon}")

        if manager.num_trajectories > 0:
            # Get stats for the first trajectory
            first_traj_stats = manager.get_trajectory_reward_stats(0)
            if first_traj_stats:
                print(f"\nStats for Trajectory 0:")
                for key, value in first_traj_stats.items():
                    print(f"  {key}: {value}")

            # Get stats for all trajectories
            all_stats = manager.get_all_trajectory_reward_stats()
            # print(f"\nAll Trajectory Stats: {all_stats}")

            # Get overall stats
            overall_stats = manager.get_overall_reward_stats()
            if overall_stats:
                print(f"\nOverall Reward Stats:")
                for key, value in overall_stats.items():
                    print(f"  {key}: {value}")
            
            # Example of getting a single trajectory's data
            states_0, actions_0, rewards_0 = manager.get_trajectory(0)
            if states_0 is not None:
                print(f"\nTrajectory 0 - States shape: {states_0.shape}, Actions shape: {actions_0.shape}, Rewards shape: {rewards_0.shape}")

    except IOError as e:
        print(e)
    except ValueError as e:
        print(f"Data validation error: {e}") 