import os
import jax
from jax import numpy as jnp
import brax
from brax.envs.base import PipelineEnv, State
from brax.generalized import pipeline
from brax.io import html, mjcf
# Removed quat_to_euler import - using safe_quat_to_euler instead to avoid CUDA cusolver errors

import mbd # For mbd.__path__[0]

# Default world boundaries for termination
WORLD_BOUND_X = 1.9
WORLD_BOUND_Y = 1.9
TARGET_REACHED_THRESHOLD_POS = 0.15 # Position threshold for reaching target
TARGET_REACHED_THRESHOLD_ORI = 0.2  # Orientation threshold (radians) for reaching target


@jax.jit
def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jax.jit
def oriented_rectangle_to_circle_distance(rect_center_xy, rect_half_extents, rect_angle, circle_center_xy, circle_radius):
    """
    Compute minimum distance between an oriented rectangle and a circle in 2D.
    
    Args:
        rect_center_xy: jnp.array of shape (2,) - rectangle center
        rect_half_extents: jnp.array of shape (2,) - rectangle half-extents [half_width, half_height]
        rect_angle: float - rectangle rotation angle
        circle_center_xy: jnp.array of shape (2,) - circle center
        circle_radius: float - circle radius
    
    Returns:
        distance: float - minimum distance between rectangle and circle (0 if overlapping)
    """
    # Generate corner points of oriented rectangle
    cos_angle = jnp.cos(rect_angle)
    sin_angle = jnp.sin(rect_angle)
    
    # Local corner offsets - vectorized
    local_corners = jnp.array([
        [-rect_half_extents[0], -rect_half_extents[1]],  # bottom-left
        [rect_half_extents[0], -rect_half_extents[1]],   # bottom-right  
        [rect_half_extents[0], rect_half_extents[1]],    # top-right
        [-rect_half_extents[0], rect_half_extents[1]]    # top-left
    ])
    
    # Rotation matrix
    rotation_matrix = jnp.array([[cos_angle, -sin_angle],
                                [sin_angle, cos_angle]])
    
    # Transform all corners to global coordinates - vectorized
    global_corners = jnp.dot(local_corners, rotation_matrix.T) + rect_center_xy
    
    # Find minimum distance from any corner to the circle center - vectorized
    corner_to_center_distances = jnp.linalg.norm(global_corners - circle_center_xy, axis=1)
    min_corner_to_center_distance = jnp.min(corner_to_center_distances)
    
    # Simple distance calculation with ReLU to ensure non-negative
    distance_rect_to_circle_surface = jax.nn.relu(min_corner_to_center_distance - circle_radius)
    
    return distance_rect_to_circle_surface



class CarEnv(PipelineEnv):
    def __init__(self, backend: str = "generalized",
                 n_frames: int = 5,
                 **kwargs):
        # Load the MuJoCo model
        xml_path = os.path.join(mbd.__path__[0], "assets/car_env.xml")
        print(f"[CarEnv Debug] Loading MJCF from: {xml_path}")
        sys = mjcf.load(xml_path)
        print(f"[CarEnv Debug] Parsed link_names: {list(sys.link_names)}")
        # print(f'system dt: {sys.dt}')
        self.init_q = jnp.array([0.0, -1.2, 0.0]) # Initial position [x, y, z_angle]

        self.target_q_xy = jnp.array([0.0, 1.25])     # Target XY position
        self.target_q_rot_z = jnp.array([0.0]) # Target Z rotation (example: 90 degrees)

        # Exact box dimensions from XML (half-extents)
        # Car: size="0.225 0.1 0.025" -> half_extents = [0.225, 0.1]
        # Obstacles: now circles with radius 0.45 (from XML: size="0.45")
        self.car_half_extents = jnp.array([0.225, 0.1])
        self.obstacle_radius = 0.45  # Circle radius from XML
        
        # Pre-compute obstacle positions to avoid recreating array every time
        self.obstacle_positions_xy = jnp.array([
            [0.7, 0.7],
            [-0.7, 0.7],
            [0.7, -0.7],
            [-0.7, -0.7],
            [0.7, 0.0],
            [-0.7, 0.0],
        ])

        super().__init__(sys=sys, backend=backend, n_frames=n_frames)

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        q = jnp.zeros(self.sys.nq).at[:3].set(self.init_q)
        # randomize q_x and q_rot
        # Split RNG for randomization
        rng_x, rng_rot = jax.random.split(rng)
        
        # Randomize initial position and rotation
        random_x = jax.random.uniform(rng_x, (), minval=-0.3, maxval=0.3)
        random_rot = jax.random.uniform(rng_rot, (), minval=-0.7, maxval=0.7)
        
        q = q.at[0].set(random_x)     # Random x position
        q = q.at[1].set(-1.2)         # Fixed y position from init_q
        q = q.at[2].set(random_rot)   # Random rotation        # Set rotation to 0.7
        qd = jnp.zeros(self.sys.qd_size())
        # reset vy to 3
        # qd = qd.at[1].set(1)
        # qd = qd.at[2].set(1)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state, jnp.zeros(self.action_size))
        done = self._get_done(pipeline_state)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        # jax.debug.print("[DEBUG] step: action={}, state.pipeline_state.qd={}", action, state.pipeline_state.qd)
        pipeline_state = self.pipeline_step(state.pipeline_state, action*jnp.array([0.5, 3.0, 3.0]))
        # jax.debug.print("[DEBUG] step: action={}, pipeline_state.qd={}", action, pipeline_state.qd)
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state, action)
        done = self._get_done(pipeline_state)
        
        # Debug prints - both before return statement
        # jax.debug.print("[DEBUG] step: reward={}, done={}", reward, done)
        # jax.debug.print("[DEBUG] car_pos: {}", pipeline_state.x.pos[0, :2])
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        """Observe car's state relative to target and its own dynamics."""
        car_joint_q = pipeline_state.q[:3]    # [x_joint_pos, y_joint_pos, z_joint_angle]
        car_joint_qd = pipeline_state.qd[:3]   # [vx_local, vy_local, wz_local]

        # Car's global pose
        car_global_pos_xy = pipeline_state.x.pos[0,:2]
        # Car's global orientation (Z-axis Euler angle)
        car_global_rot_quat = pipeline_state.x.rot[0]
        car_global_euler_z = pipeline_state.q[2]

        # Relative position to target
        rel_pos_to_target_xy = self.target_q_xy - car_global_pos_xy
        
        # Relative orientation to target (difference in Z rotation) - vectorized
        target_rot_z_norm = normalize_angle(self.target_q_rot_z[0])
        car_rot_z_norm = normalize_angle(car_global_euler_z)
        rel_rot_to_target_z = normalize_angle(target_rot_z_norm - car_rot_z_norm)

        obs = jnp.concatenate([
            car_joint_q,              # 3 (local pose)
            car_joint_qd,             # 3 (local velocities)
            rel_pos_to_target_xy,     # 2
            jnp.array([rel_rot_to_target_z]) # 1
        ])
        
        # JAX-compatible debug: Print observations without conditional logic
        # jax.debug.print("[DEBUG] observations: car_pos={}, car_rot={}, obs_shape={}", 
        #                car_global_pos_xy, car_global_euler_z, obs.shape)

        return obs

    def _get_reward(self, pipeline_state: pipeline.State, applied_motor_action: jnp.ndarray) -> jnp.ndarray:
        """Calculates the reward for the current state and action."""
        car_global_pos_xy = pipeline_state.x.pos[0,:2]

        car_global_euler_z = pipeline_state.q[2]

        # JAX-compatible debug: Print inputs without conditional logic
        # jax.debug.print("[DEBUG] reward inputs: car_pos={}, car_rot={}", 
        #                car_global_pos_xy, car_global_euler_z)

        # 1. Target Proximity Reward (dense reward)
        dist_to_target_xy = jnp.linalg.norm(self.target_q_xy - car_global_pos_xy)
        # clip dist_to_target_xy to be at least 0.1
        reward_target_dist = -dist_to_target_xy

        # 2. Target Orientation Reward (dense reward) - using vectorized normalize_angle

        diff_rot_z = normalize_angle(self.target_q_rot_z[0] - car_global_euler_z)
        reward_target_orient = -jnp.abs(diff_rot_z)

        # JAX-compatible debug: Print target rewards
        # jax.debug.print("[DEBUG] target rewards: dist={}, orient={}", 
        #                reward_target_dist, reward_target_orient)

        # 3. Obstacle Avoidance Penalty using rectangle-to-circle distance
        # Use pre-computed obstacle positions instead of recreating array
        
        # 4. Target Velocity Reward (dense reward)
        velocity_reward = -jnp.sum(jnp.abs(pipeline_state.qd[:2]))

        # Calculate distances to all obstacles - vectorized with vmap
        distances_to_obstacles = jax.vmap(
            lambda obstacle_center: oriented_rectangle_to_circle_distance(
                car_global_pos_xy, self.car_half_extents, car_global_euler_z,
                obstacle_center, self.obstacle_radius
            )
        )(self.obstacle_positions_xy)
        
        # Find the minimum distance to any obstacle
        min_distance_to_obstacles = jnp.min(distances_to_obstacles)
        safety_margin = 0.10  # Start applying penalty when closer than this distance
        
        # Apply proximity penalty for being near obstacle
        obstacle_penalty = -jnp.power(jax.nn.relu(safety_margin - min_distance_to_obstacles), 2)



        # 4. Control Cost - vectorized
        ctrl_cost = -0.1 * jnp.sum(jnp.square(applied_motor_action[:2]))

        # 5. Goal Reaching Reward
        orient_to_target_ok = jnp.abs(diff_rot_z) < TARGET_REACHED_THRESHOLD_ORI
        reached_target = (dist_to_target_xy < TARGET_REACHED_THRESHOLD_POS) & orient_to_target_ok
        reward_goal_reached = 30.0 * reached_target

        # JAX-compatible debug: Print reward components
        # jax.debug.print("[DEBUG] reward components: obstacle_penalty={}, ctrl_cost={}", 
        #                penalty_obstacle, ctrl_cost)

        total_reward = (
            2.0 * reward_target_dist +      # Weight for distance
            0.0 * reward_target_orient +    # Weight for orientation
            20.0 * obstacle_penalty +        # Weight for obstacle penalty
            1.0 * velocity_reward +          # Weight for velocity reward
            1.0 * ctrl_cost +                # Weight for control cost
            reward_goal_reached
        )
        
        # JAX-compatible debug: Print final reward
        # jax.debug.print("[DEBUG] total_reward: {}", obstacle_penalty)
        # jax.debug.print("target_dist: {reward_target_dist:.2f}, target_orient: {reward_target_orient:.2f}, obstacle_penalty: {obstacle_penalty:.2f}, velocity_reward: {velocity_reward:.2f}, ctrl_cost: {ctrl_cost:.2f}, goal_reaching{reward_goal_reached:.2f}",
        #                 reward_target_dist=reward_target_dist,
        #                 reward_target_orient=reward_target_orient, 
        #                 obstacle_penalty=obstacle_penalty,
        #                 velocity_reward=velocity_reward,
        #                 ctrl_cost=ctrl_cost,
        #                 reward_goal_reached=reward_goal_reached)

        return total_reward.astype(jnp.float32)

    def _get_done(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        """Checks if the episode is done."""
        car_global_pos_xy = pipeline_state.x.pos[0,:2]
        car_global_rot_quat = pipeline_state.x.rot[0]
        car_global_euler_z = pipeline_state.q[2]

        # 1. Reached Target - using vectorized normalize_angle
        dist_to_target_xy = jnp.linalg.norm(self.target_q_xy - car_global_pos_xy)
        target_rot_z_norm = normalize_angle(self.target_q_rot_z[0])
        car_rot_z_norm = normalize_angle(car_global_euler_z)
        diff_rot_z = normalize_angle(target_rot_z_norm - car_rot_z_norm)
        orient_to_target_ok = jnp.abs(diff_rot_z) < TARGET_REACHED_THRESHOLD_ORI
        reached_target = (dist_to_target_xy < TARGET_REACHED_THRESHOLD_POS) & orient_to_target_ok

        # 2. Obstacle Collision using rectangle-to-circle distance
        # Use pre-computed obstacle positions instead of recreating array
        
        # # Calculate distances to all obstacles - vectorized with vmap
        distances_to_obstacles = jax.vmap(
            lambda obstacle_center: oriented_rectangle_to_circle_distance(
                car_global_pos_xy, self.car_half_extents, car_global_euler_z,
                obstacle_center, self.obstacle_radius
            )
        )(self.obstacle_positions_xy)
        
        # Collision if any obstacle distance is 0 (touching/overlapping)
        min_distance_to_obstacles = jnp.min(distances_to_obstacles)
        collided_obstacle = min_distance_to_obstacles <= 0.0

        # 3. Out of Bounds - vectorized comparison
        out_of_bounds = jnp.any(jnp.abs(car_global_pos_xy) > jnp.array([WORLD_BOUND_X, WORLD_BOUND_Y]))
        
        done = reached_target | collided_obstacle | out_of_bounds
        
        
        # JAX-compatible debug: Print done condition
        # jax.debug.print("[DEBUG] done condition: target={}, collision={}, bounds={}, final={}", 
        #                reached_target, collided_obstacle, out_of_bounds, done)
        
        return done.astype(jnp.float32)

    @property
    def action_size(self):
        return 3 # target_local_vx, target_local_vy, target_local_wz

    @property
    def observation_size(self):
        # car_joint_q (3) + car_joint_qd (3) + rel_pos_to_target_xy (2) + rel_rot_to_target_z (1)
        return 3 + 3 + 2 + 1 # = 9


# Main function for testing
def main():
    env = CarEnv()
    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step) # JIT compile for speed
    env_reset = jax.jit(env.reset) # JIT compile for speed
    # env_step = env.step
    # env_reset = env.reset

    rng_env,rng = jax.random.split(rng)
    state = env_reset(rng_env)
    rollout = [state.pipeline_state]
    print(f"Initial obs: {state.obs}")
    print(f"Initial reward: {state.reward}")
    print(f"Step 0: Obs: {state.obs}, Reward: {state.reward:.2f}, Done: {state.done}")
    # print qd
    jax.debug.print("[DEBUG] initial qd: {}", state.pipeline_state.qd)
    for i in range(50): # Increased steps for testing
        rng, rng_act = jax.random.split(rng)
        # Random actions for now to test dynamics
        act = jnp.array([0, 1.0, 0.0])

        state = env_step(state, act)
        rollout.append(state.pipeline_state)
        # print(f"Step {i}: Obs: {state.obs}, Reward: {state.reward:.2f}, Done: {state.done}")
        jax.debug.print("[DEBUG] step: reward={}, done={}", state.reward, state.done)
        if state.done:
            print(f"Episode finished at step {i+1}. Reward: {state.reward:.2f}, Done: {state.done}")
            break
        # import pdb; pdb.set_trace()
    
    if not state.done:
        print(f"Episode did not finish after {len(rollout)-1} steps.")


    webpage = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout) # Use env.sys.replace(dt=env.dt) as per PushT
    path = f"{mbd.__path__[0]}/../results/car_env"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/vis.html", "w") as f:
        f.write(webpage)
    print(f"Visualization saved to {path}/vis.html")


if __name__ == "__main__":
    main() 