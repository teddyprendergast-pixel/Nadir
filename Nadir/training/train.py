import os
import sys

# Auto-relaunch using the project's virtual environment if invoked with a different Python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

if "XLA_FLAGS" not in os.environ:
    num_threads = str(os.cpu_count() or 20)
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={num_threads}"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 20)

import argparse
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from datetime import datetime

try:
    from tqdm import tqdm
    def tprint(*args, **kwargs):
        tqdm.write(" ".join(map(str, args)), **kwargs)
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    tprint = print

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from .config import PPOConfig, EnvConfig
    from .networks import create_actor_critic
    from .ppo import PPOTrainer, RunnerState
except ImportError:
    from nadir.training.config import PPOConfig, EnvConfig
    from nadir.training.networks import create_actor_critic
    from nadir.training.ppo import PPOTrainer, RunnerState

from nadir.sim.env_mjx import NadirEnv

def main():
    parser = argparse.ArgumentParser(description="Train Nadir bipedal robot policy")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=24, help="Number of steps per rollout")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup configs
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    env_config = EnvConfig(num_envs=args.num_envs)
    
    # Initialize random keys
    rng = jax.random.PRNGKey(ppo_config.seed)
    rng, rng_init, rng_env = jax.random.split(rng, 3)
    
    # Initialize real MJX Environment
    env = NadirEnv(num_envs=args.num_envs)
    env_state, obs, priv_obs = env.reset(jax.random.split(rng_env, args.num_envs))
    
    # Initialize Network
    actor_critic = create_actor_critic(ppo_config)
    dummy_obs = jnp.zeros((env_config.num_envs, env_config.obs_dim))
    dummy_priv_obs = jnp.zeros((env_config.num_envs, env_config.privileged_obs_dim))
    
    params = actor_critic.init(rng_init, dummy_obs, dummy_priv_obs)
    
    # Initialize Trainer
    trainer = PPOTrainer(ppo_config, env, actor_critic)
    opt_state = trainer.optimizer.init(params)
    
    runner_state = RunnerState(
        params=params,
        opt_state=opt_state,
        env_state=env_state,
        obs=obs,
        privileged_obs=priv_obs,
        rng=rng
    )
    
    # Setup Checkpointer
    ckpt_dir = os.path.join(ppo_config.checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt_state = checkpointer.restore(args.resume)
        runner_state = runner_state._replace(
            params=ckpt_state['params'],
            opt_state=ckpt_state['opt_state']
        )
    
    # Calculate updates
    num_updates = ppo_config.total_timesteps // (ppo_config.num_steps * ppo_config.num_envs)
    
    print(f"Starting training for {num_updates} updates...")
    
    # Wrap train step with lax.scan across total updates (could batch this if needed)
    import time
    for update in tqdm(range(1, num_updates + 1), desc="Training", unit="update"):
        update_start_time = time.time()
        runner_state, metrics = trainer.train_step(runner_state, None)
        
        # Block until computation is complete to get accurate timing
        metrics['reward_sum'].block_until_ready()
        update_time = time.time() - update_start_time
        
        if update % ppo_config.log_interval == 0:
            sps = (ppo_config.num_steps * ppo_config.num_envs) / update_time if update_time > 0 else 0
            tprint(f"Update: {update}/{num_updates}")
            tprint(f"Reward Sum: {metrics['reward_sum']:.2f}")
            tprint(f"Policy Loss: {metrics['policy_loss']:.4f}")
            tprint(f"Value Loss: {metrics['value_loss']:.4f}")
            tprint(f"SPS: {sps:.0f}")
            tprint("-" * 30)
            
        if update % ppo_config.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"update_{update}")
            checkpointer.save(os.path.abspath(ckpt_path), {
                'params': runner_state.params,
                'opt_state': runner_state.opt_state
            })
            tprint(f"Saved checkpoint to {ckpt_path}")

    try:
        checkpointer.wait_until_finished()
        checkpointer.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
