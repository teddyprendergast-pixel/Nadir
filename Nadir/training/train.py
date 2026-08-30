"""PPO training entry point.

Run from the repository root:

    python -m Nadir.training.train --num-envs 4096 --total-timesteps 300000000

The import below is relative on purpose. The previous version did
`from nadir.sim.env_mjx import NadirEnv` — lowercase — which resolves on a
case-insensitive macOS filesystem and is a hard ImportError on any Linux
cluster. It had never been run on the machine it was written for.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from ..sim.env_mjx import NadirEnv
from .config import EnvConfig, PPOConfig
from .networks import create_actor_critic
from .ppo import PPOTrainer, RunnerState

METRIC_FIELDS = [
    "update", "env_steps", "wall_time_s", "sps",
    "episode_return", "episode_length", "episodes_finished", "termination_rate",
    "mean_step_reward", "loss", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_frac",
]


def main():
    p = argparse.ArgumentParser(description="Train the Nadir walking policy")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=24)
    p.add_argument("--total-timesteps", type=int, default=300_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=100)
    args = p.parse_args()

    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    env_config = EnvConfig(num_envs=args.num_envs)

    print("JAX devices:", jax.devices(), flush=True)

    rng = jax.random.PRNGKey(ppo_config.seed)
    rng, rng_init, rng_env = jax.random.split(rng, 3)

    env = NadirEnv(num_envs=args.num_envs, config=env_config.env_kwargs())
    env_state, obs, priv_obs = env.reset(jax.random.split(rng_env, args.num_envs))

    # Assert the config's advertised dimensions match what the env actually
    # produces, rather than trusting two numbers to stay in agreement by hand.
    assert obs.shape[-1] == env_config.obs_dim, (obs.shape, env_config.obs_dim)
    assert priv_obs.shape[-1] == env_config.privileged_obs_dim, (
        priv_obs.shape, env_config.privileged_obs_dim
    )

    actor_critic = create_actor_critic(ppo_config)
    params = actor_critic.init(
        rng_init,
        jnp.zeros((1, env_config.obs_dim)),
        jnp.zeros((1, env_config.privileged_obs_dim)),
    )

    trainer = PPOTrainer(ppo_config, env, actor_critic)
    opt_state = trainer.optimizer.init(params)

    runner_state = RunnerState(
        params=params, opt_state=opt_state, env_state=env_state,
        obs=obs, privileged_obs=priv_obs, rng=rng,
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.abspath(os.path.join(ppo_config.checkpoint_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {"ppo": ppo_config.__dict__, "env": env_config.__dict__, "argv": vars(args)},
            f, indent=2, default=str,
        )

    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        ckpt = checkpointer.restore(os.path.abspath(args.resume))
        runner_state = runner_state._replace(
            params=ckpt["params"], opt_state=ckpt["opt_state"]
        )

    steps_per_update = ppo_config.num_steps * ppo_config.num_envs
    num_updates = ppo_config.total_timesteps // steps_per_update

    metrics_path = os.path.join(run_dir, "metrics.csv")
    mf = open(metrics_path, "w", newline="")
    writer = csv.DictWriter(mf, fieldnames=METRIC_FIELDS)
    writer.writeheader()

    print(
        f"{num_updates} updates x {steps_per_update:,} steps "
        f"= {num_updates * steps_per_update:,} env steps",
        flush=True,
    )
    print(f"Run directory: {run_dir}", flush=True)

    best_return = -jnp.inf
    t0 = time.time()

    for update in range(1, num_updates + 1):
        runner_state, metrics = trainer.train_step(runner_state, None)

        if update % ppo_config.log_interval == 0 or update == 1:
            m = {k: float(v) for k, v in metrics.items()}
            env_steps = update * steps_per_update
            wall = time.time() - t0
            row = {
                "update": update, "env_steps": env_steps,
                "wall_time_s": round(wall, 1), "sps": int(env_steps / max(wall, 1e-9)),
            }
            row.update({k: m.get(k) for k in METRIC_FIELDS if k in m})
            writer.writerow(row)
            mf.flush()
            print(
                f"[{update:>5}/{num_updates}] steps={env_steps/1e6:6.1f}M "
                f"ret={m['episode_return']:8.2f} len={m['episode_length']:7.1f} "
                f"term={m['termination_rate']:.2f} "
                f"vloss={m['value_loss']:8.3f} kl={m['approx_kl']:.4f} "
                f"sps={row['sps']:,}",
                flush=True,
            )

            # Keep the best policy by mean episode return, not just the last.
            if m["episodes_finished"] > 0 and m["episode_return"] > best_return:
                best_return = m["episode_return"]
                checkpointer.save(
                    os.path.join(run_dir, "best"),
                    {"params": runner_state.params, "opt_state": runner_state.opt_state},
                    force=True,
                )

        if update % ppo_config.save_interval == 0:
            checkpointer.save(
                os.path.join(run_dir, f"update_{update}"),
                {"params": runner_state.params, "opt_state": runner_state.opt_state},
            )

    checkpointer.save(
        os.path.join(run_dir, "final"),
        {"params": runner_state.params, "opt_state": runner_state.opt_state},
        force=True,
    )
    mf.close()
    # Orbax saves asynchronously. Without this the interpreter can start
    # shutting down while a commit thread is still queueing work, which
    # surfaces as "cannot schedule new futures after interpreter shutdown"
    # and can leave the final checkpoint half-written.
    checkpointer.wait_until_finished()
    checkpointer.close()
    print(f"Done. Best mean episode return: {float(best_return):.2f}", flush=True)
    print(f"Checkpoints and metrics in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
