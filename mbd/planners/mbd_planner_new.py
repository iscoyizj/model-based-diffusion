import jax
import os
from jax import numpy as jnp
# Replaced InterpolatedUnivariateSpline with jnp.interp to avoid CUDA cusolver_getrf_ffi error
from tqdm import tqdm
import time
import functools
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt
from brax.io import html
import scienceplots
import mbd
import numpy as np  # Added for Zarr
import zarr  # Added for Zarr
from jax import tree_util

plt.style.use("science")

# # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
# xla_flags = os.environ.get("XLA_FLAGS", "")
# xla_flags += " --xla_gpu_triton_gemm_any=True"
# os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state, state.obs)

    states, (rews, pipline_states, obs) = jax.lax.scan(step, state, us)
    # jax.debug.print("[DEBUG] rollout_us: states.obs.shape={}", states.obs.shape)
    # obs = states.obs
    return rews, pipline_states, obs


@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    # env
    env_name: str = "car_env"
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 60  # horizon of samples
    Hnode: int = 8  # node number for control
    Ndiffuse: int = 9  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    horizon_diffuse_factor: float = 0.9  # factor to scale the sigma of horizon diffuse
    traj_diffuse_factor: float = 0.5  # factor to scale the sigma of trajectory diffuse
    # data saving
    save_data: bool = True  # whether to save trajectory data
    data_output_dir: str = "../results/"+env_name  # directory to save data
    max_trajectories_to_save: int = 800  # maximum number of best trajectories to save

class MBDPI:
    def __init__(self, args: Args, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        # Use simple linear interpolation instead of splines to avoid CUDA issues
        us = jnp.interp(self.step_us, self.step_nodes, nodes)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        # Use simple linear interpolation instead of splines to avoid CUDA issues
        nodes = jnp.interp(self.step_nodes, self.step_us, us)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)

        # esitimate mu_0tm1
        rewss, pipeline_statess, obs = self.rollout_us_vmap(state, us)
        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i)  / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        info = {
            "rews": rewss,
            "obs": obs,
            "pipeline_statess": pipeline_statess,
            "us": us,
        }

        return rng, Ybar, info

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, info = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode + 1)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{info['rews'].mean():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def save_trajectory_data(args: Args, rewss, pipeline_statess, us, state_init, env):
    """Save trajectory data to Zarr format, similar to PPO data collection."""
    if not args.save_data:
        return
    
    print("Saving trajectory data to Zarr format...")
    
    # Create output directory
    data_output_dir = f"../results/{args.env_name}"
    os.makedirs(data_output_dir, exist_ok=True)
    
    # Create Zarr file
    zarr_file_path = os.path.join(data_output_dir, f"mbd_trajectories_{time.strftime('%Y%m%d_%H%M%S')}.zarr")
    
    try:
        zroot = zarr.open_group(zarr_file_path, "w")
        zdata = zroot.create_group("data")
        zmeta = zroot.create_group("meta")
        
        # Determine dimensions
        obs_dim = env.observation_size
        action_dim = env.action_size
        
        print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
        print(f"Rewards shape: {rewss.shape}, Pipeline states q shape: {pipeline_statess.q.shape}")
        print(f"Actions shape: {us.shape}")
        
        # Select best trajectories based on mean reward
        trajectory_rewards = rewss.mean(axis=1)  # Mean reward per trajectory
        best_indices = jnp.argsort(trajectory_rewards)[-args.max_trajectories_to_save:]  # Top trajectories
        
        print(f"Selected {len(best_indices)} best trajectories out of {len(trajectory_rewards)} total")
        print(f"Best trajectory rewards: {trajectory_rewards[best_indices]}")
        
        # Extract best trajectories
        best_rewss = rewss[best_indices]  # Shape: (num_best, Hsample)
        best_us = us[best_indices]  # Shape: (num_best, Hsample, action_dim)
        
        # Extract pipeline states for best trajectories
        def extract_best_pipeline_states(x):
            return x[best_indices]
        best_pipeline_statess = tree_util.tree_map(extract_best_pipeline_states, pipeline_statess)
        
        # Convert to observations for each trajectory
        best_obs_list = []
        for traj_idx in range(len(best_indices)):
            # Extract single trajectory pipeline states
            def extract_single_trajectory(x):
                return x[traj_idx]
            single_traj_states = tree_util.tree_map(extract_single_trajectory, best_pipeline_statess)
            
            # Convert each timestep to observation
            traj_obs = []
            for t in range(single_traj_states.q.shape[0]):
                def extract_timestep(x):
                    return x[t]
                timestep_state = tree_util.tree_map(extract_timestep, single_traj_states)
                obs = env._get_obs(timestep_state)
                traj_obs.append(obs)
            
            best_obs_list.append(jnp.stack(traj_obs))  # Shape: (Hsample, obs_dim)
        
        best_obs_array = jnp.stack(best_obs_list)  # Shape: (num_best, Hsample, obs_dim)
        
        # Flatten data for storage (similar to PPO approach)
        # Reshape from (num_traj, horizon, dim) to (num_traj * horizon, dim)
        num_best_trajectories = best_obs_array.shape[0]
        horizon = best_obs_array.shape[1]
        total_steps = num_best_trajectories * horizon
        
        flat_obs = best_obs_array.reshape(total_steps, obs_dim)
        flat_actions = best_us.reshape(total_steps, action_dim)
        flat_rewards = best_rewss.reshape(total_steps)
        
        # Convert to numpy for Zarr storage
        flat_obs_np = np.array(flat_obs, dtype=np.float32)
        flat_actions_np = np.array(flat_actions, dtype=np.float32)
        flat_rewards_np = np.array(flat_rewards, dtype=np.float32)
        
        # Create Zarr datasets
        zdata.create_dataset("state", data=flat_obs_np, dtype='float32')
        zdata.create_dataset("action", data=flat_actions_np, dtype='float32')
        zdata.create_dataset("reward", data=flat_rewards_np, dtype='float32')
        
        # Create episode end markers (end of each trajectory)
        episode_ends = np.arange(horizon, total_steps + 1, horizon, dtype=np.int64)
        zmeta.create_dataset("episode_ends", data=episode_ends, dtype='int64')
        
        # Save metadata
        zmeta.attrs['num_trajectories'] = num_best_trajectories
        zmeta.attrs['horizon'] = horizon
        zmeta.attrs['obs_dim'] = obs_dim
        zmeta.attrs['action_dim'] = action_dim
        zmeta.attrs['env_name'] = args.env_name
        zmeta.attrs['total_steps'] = total_steps
        
        print(f"Successfully saved {total_steps} steps from {num_best_trajectories} trajectories to {zarr_file_path}")
        print(f"Episode end markers: {episode_ends.tolist()}")
        print("Dataset structure:")
        print(zroot.tree())
        
    except Exception as e:
        print(f"Error saving trajectory data: {e}")
        import traceback
        traceback.print_exc()


def main(args: Args):
    rng = jax.random.PRNGKey(seed=args.seed)

    env = mbd.envs.get_env(args.env_name)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(args, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([args.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = 200
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    
    Y0 = YN
    Ybars = []
    with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
        for i in pbar:
            rng, Y0, info = mbdpi.reverse_once(state_init, rng, Y0, mbdpi.sigma_control)
            Ybars.append(Y0)
            # Update the progress bar's suffix to show the current reward
            pbar.set_postfix({"rew": f"{info['rews'].mean():.2e}"})

    # Convert the final optimized control nodes to action sequences
    us_final = mbdpi.node2u_vmap(Y0)  # This gives us a single trajectory
    
    # But for data saving, we need all the sampled trajectories from the last diffusion step
    # Let's run one more reverse_once to get all the sampled action sequences
    print("Generating final sampled trajectories for data saving...")
    rng, Y0_final, final_info = mbdpi.reverse_once(state_init, rng, Y0, mbdpi.sigma_control)
    
    # Now run rollout to get all trajectories
    # We need to reconstruct the sampled Y0s from the last iteration
    # Generate samples like in reverse_once
    rng, Y0s_rng = jax.random.split(rng)
    eps_Y = jax.random.normal(
        Y0s_rng, (args.Nsample, args.Hnode + 1, mbdpi.nu)
    )
    Y0s = eps_Y * mbdpi.sigma_control[None, :, None] + Y0
    # append Y0s with Y0 to also evaluate Y0
    Y0s = jnp.concatenate([Y0s, Y0[None]], axis=0)
    Y0s = jnp.clip(Y0s, -1.0, 1.0)
    
    # Convert all control node sequences to action sequences
    us = mbdpi.node2u_vvmap(Y0s)  # Shape: (Nsample+1, Hsample, action_dim)
    
    print(f"Generated action sequences with shape: {us.shape}")
    print(f"Expected shape: ({args.Nsample + 1}, {args.Hsample}, {mbdpi.nu})")

    # host webpage with flask
    import flask

    # esitimate mu_0tm1
    rewss, pipeline_statess = mbdpi.rollout_us_vmap(state_init, us)
    
    print(f"After rollout - Rewards shape: {rewss.shape}, Expected: ({args.Nsample + 1}, {args.Hsample})")
    print(f"After rollout - Pipeline states q shape: {pipeline_statess.q.shape}")
    
    # Check if environment is terminating early
    # Let's check the done flags for the first trajectory
    if hasattr(pipeline_statess, 'done'):
        print(f"Done flags shape: {pipeline_statess.done.shape}")
        first_traj_done = pipeline_statess.done[0]  # First trajectory
        print(f"First trajectory done flags: {first_traj_done}")
        print(f"Environment terminates at step: {jnp.argmax(first_traj_done) if jnp.any(first_traj_done) else 'Never'}")
    
    # Save trajectory data to Zarr
    save_trajectory_data(args, rewss, pipeline_statess, us, state_init, env)
    
    # Select the best trajectory for visualization
    # rewss shape: (Nsample+1, Hsample)
    # pipeline_statess is a batched pipeline state with shape (Nsample+1, Hsample) for each attribute
    
    # Option 1: Select the trajectory with highest mean reward
    trajectory_rewards = rewss.mean(axis=1)  # Mean reward per trajectory
    best_trajectory_idx = jnp.argmax(trajectory_rewards)
    
    # Option 2: Use the last trajectory (which is the mean/reference trajectory)
    # best_trajectory_idx = -1
    
    print(f"Selected trajectory {best_trajectory_idx} with mean reward: {trajectory_rewards[best_trajectory_idx]:.3f}")
    print(f"Pipeline states type: {type(pipeline_statess)}")
    print(f"Pipeline states q shape: {pipeline_statess.q.shape}")
    print(f"Total samples evaluated: {len(trajectory_rewards)}")
    
    # Extract the best trajectory pipeline states using JAX tree operations
    # We need to slice each component of the pipeline state
    def extract_trajectory(x):
        """Extract a single trajectory from batched data."""
        return x[best_trajectory_idx]
    
    # Apply the extraction to the entire pipeline state tree
    best_trajectory_pipeline_states = tree_util.tree_map(extract_trajectory, pipeline_statess)
    
    # Now convert to a list of individual pipeline states for each timestep
    # best_trajectory_pipeline_states has shape (Hsample,) for each attribute
    rollout = []
    for t in range(best_trajectory_pipeline_states.q.shape[0]):  # Hsample timesteps
        # Extract timestep t from the trajectory
        def extract_timestep(x):
            return x[t]
        
        timestep_state = tree_util.tree_map(extract_timestep, best_trajectory_pipeline_states)
        rollout.append(timestep_state)
    
    print(f"Rollout length: {len(rollout)}")
    print(f"First rollout state q shape: {rollout[0].q.shape}")
    
    app = flask.Flask(__name__)
    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)

    # Save visualization to file for easy access
    os.makedirs("../results", exist_ok=True)
    with open("../results/car_env/diffusion_rollout.html", "w") as f:
        f.write(webpage)
    print("Visualization saved to ../results/car_env/diffusion_rollout.html")



if __name__ == "__main__":
    main(tyro.cli(Args))
