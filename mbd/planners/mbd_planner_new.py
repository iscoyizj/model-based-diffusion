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

plt.style.use("science")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    # env
    env_name: str = "car_env"
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 80  # horizon of samples
    Hnode: int = 16  # node number for control
    Ndiffuse: int = 10  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    horizon_diffuse_factor: float = 0.9  # factor to scale the sigma of horizon diffuse
    traj_diffuse_factor: float = 0.5  # factor to scale the sigma of trajectory diffuse

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
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
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
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
        }

        return rng, Ybar, info

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode + 1)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
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
    # with tqdm(range(Nstep), desc="Rollout") as pbar:
    #     for t in pbar:
    #         # forward single step
    #         state = step_env(state, Y0[0])
    #         rollout.append(state.pipeline_state)
    #         rews.append(state.reward)
    #         us.append(Y0[0])

    #         # update Y0
    #         Y0 = mbdpi.shift(Y0)

    #         t0 = time.time()
    #         for i in range(args.Ndiffuse):
    #             rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, mbdpi.sigma_control*(args.traj_diffuse_factor**i))
    #         rews_plan.append(info["rews"].mean())
    #         freq = 1 / (time.time() - t0)
    #         pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    # rew = jnp.array(rews).mean()
    # print(f"mean reward = {rew:.2e}")
    
    Y0 = YN
    Ybars = []
    with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
        for i in pbar:
            rng, Y0, info = mbdpi.reverse_once(state_init, rng, Y0, mbdpi.sigma_control)
            Ybars.append(Y0)
            # Update the progress bar's suffix to show the current reward
            # jax.debug.print("[DEBUG] state: {state.pipeline_state.x.pos[0, :2]}")
            pbar.set_postfix({"rew": f"{info['rews'].mean():.2e}"})


    # webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)

    us = mbdpi.node2u_vmap(Y0)
    # # save us



    # # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig("./results/rews_plan.png")

    # # host webpage with flask
    render_us = functools.partial(
        mbd.utils.render_us,
        step_env,
        env.sys.tree_replace({"opt.timestep": env.dt}),
    )
    webpage = render_us(state_init, us)
    path = f"{mbd.__path__[0]}/../results/{args.env_name}"
    with open(f"{path}/rollout.html", "w") as f:
        f.write(webpage)
    rewss_final, _ = mbdpi.rollout_us(state_init, us)

    # @app.route("/")
    # def index():
    #     return webpage

    # app.run(port=5000)


if __name__ == "__main__":
    main(tyro.cli(Args))
