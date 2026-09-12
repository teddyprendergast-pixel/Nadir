import os
import sys

# Pre-parse --num-devices before importing JAX so XLA flags can be set
num_devices = 16
for i, arg in enumerate(sys.argv):
    if arg == "--num-devices" and i + 1 < len(sys.argv):
        num_devices = int(sys.argv[i + 1])
    elif arg.startswith("--num-devices="):
        num_devices = int(arg.split("=")[1])

existing_xla = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in existing_xla:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices} " + existing_xla

import argparse
import time
import jax
import jax.numpy as jnp
import flax.jax_utils as flax_utils
import optax
import orbax.checkpoint as ocp
from datetime import datetime

try:
    from tqdm import tqdm
    def tprint(*args, **kwargs):
        tqdm.write(" ".join(map(str, args)), **kwargs)
        sys.stdout.flush()
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    def tprint(*args, **kwargs):
        print(*args, **kwargs, flush=True)

from .config import PPOConfig, EnvConfig
from .networks import create_actor_critic
from .ppo_cpu import PPOTrainerCPU, RunnerState
from nadir.sim.env_mjx import NadirEnv

def main():
    parser = argparse.ArgumentParser(description="CPU-optimized training for Nadir bipedal robot policy")
    parser.add_argument("--num-devices", type=int, default=num_devices, help="Number of virtual CPU devices to shard across")
    parser.add_argument("--num-envs", type=int, default=128, help="Total number of parallel environments across all devices (default: 128 for CPU cache)")
    parser.add_argument("--num-steps", type=int, default=24, help="Number of steps per rollout")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches per PPO epoch (default: 4 for CPU)")
    parser.add_argument("--solver", type=str, default="CG", choices=["CG", "Newton"], help="MuJoCo solver (default: CG)")
    parser.add_argument("--iterations", type=int, default=12, help="MuJoCo solver iterations (default: 12)")
    parser.add_argument("--decimation", type=int, default=5, help="Physics steps per policy step (default: 5 for 2x faster physics at 50Hz policy)")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N updates (default: 5)")
    parser.add_argument("--save-interval", type=int, default=100, help="Save every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    devs = jax.devices()
    actual_num_devices = len(devs)
    if actual_num_devices < args.num_devices:
        print(f"Warning: requested {args.num_devices} devices, but JAX initialized with {actual_num_devices}.")
        active_devices = actual_num_devices
    else:
        active_devices = args.num_devices
        
    if args.num_envs % active_devices != 0:
        raise ValueError(f"--num-envs ({args.num_envs}) must be divisible by active devices ({active_devices})")
        
    envs_per_device = args.num_envs // active_devices
    print(f"Using {active_devices} CPU devices, {envs_per_device} environments per device (total {args.num_envs} envs).")
    print(f"Solver: {args.solver} ({args.iterations} iters), Decimation: {args.decimation}, Minibatches: {args.num_minibatches}")
    
    # Setup configs
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    env_config = EnvConfig(num_envs=envs_per_device)
    
    # Initialize random keys
    rng = jax.random.PRNGKey(ppo_config.seed)
    rng, rng_init, rng_env = jax.random.split(rng, 3)
    
    # Initialize environment per device with solver tuning
    env = NadirEnv(
        num_envs=envs_per_device,
        config={"solver": args.solver, "iterations": args.iterations, "decimation": args.decimation}
    )
    
    # Multi-device reset
    @jax.pmap
    def pmap_reset(rng_key):
        subkeys = jax.random.split(rng_key, envs_per_device)
        return env.reset(subkeys)
        
    device_reset_keys = jax.random.split(rng_env, active_devices)
    env_state, obs, priv_obs = pmap_reset(device_reset_keys)
    
    # Initialize Network
    actor_critic = create_actor_critic(ppo_config)
    dummy_obs = jnp.zeros((envs_per_device, env_config.obs_dim))
    dummy_priv_obs = jnp.zeros((envs_per_device, env_config.privileged_obs_dim))
    params = actor_critic.init(rng_init, dummy_obs, dummy_priv_obs)
    
    # Initialize CPU Trainer
    trainer = PPOTrainerCPU(ppo_config, env, actor_critic, num_devices=active_devices)
    opt_state = trainer.optimizer.init(params)
    
    # Replicate model weights and optimizer state across devices
    replicated_params = flax_utils.replicate(params)
    replicated_opt_state = flax_utils.replicate(opt_state)
    device_runner_rngs = jax.random.split(rng, active_devices)
    
    runner_state = RunnerState(
        params=replicated_params,
        opt_state=replicated_opt_state,
        env_state=env_state,
        obs=obs,
        privileged_obs=priv_obs,
        rng=device_runner_rngs
    )
    
    # Setup Checkpointer
    ckpt_dir = os.path.join(ppo_config.checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt_state = checkpointer.restore(args.resume)
        replicated_resume_params = flax_utils.replicate(ckpt_state['params'])
        replicated_resume_opt = flax_utils.replicate(ckpt_state['opt_state'])
        runner_state = runner_state._replace(
            params=replicated_resume_params,
            opt_state=replicated_resume_opt
        )
        
    # Calculate updates
    num_updates = ppo_config.total_timesteps // (ppo_config.num_steps * ppo_config.num_envs)
    print(f"Starting CPU-parallel training for {num_updates} updates across {active_devices} CPU devices...")
    
    # Compile pmap train step
    pmap_train_step = jax.pmap(trainer.train_step, axis_name="devices")
    
    for update in tqdm(range(1, num_updates + 1), desc="Training CPU", unit="update"):
        update_start_time = time.time()
        runner_state, metrics = pmap_train_step(runner_state, None)
        
        # Block until computation is complete
        metrics['reward_sum'].block_until_ready()
        update_time = time.time() - update_start_time
        
        if update % ppo_config.log_interval == 0:
            sps = (ppo_config.num_steps * ppo_config.num_envs) / update_time if update_time > 0 else 0
            tprint(f"Update: {update}/{num_updates}")
            tprint(f"Reward Sum: {float(metrics['reward_sum'][0]):.2f}")
            tprint(f"Policy Loss: {float(metrics['policy_loss'][0]):.4f}")
            tprint(f"Value Loss: {float(metrics['value_loss'][0]):.4f}")
            tprint(f"SPS: {sps:.0f}")
            tprint("-" * 30)
            
        if update % ppo_config.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"update_{update}")
            unreplicated_params = flax_utils.unreplicate(runner_state.params)
            unreplicated_opt = flax_utils.unreplicate(runner_state.opt_state)
            checkpointer.save(os.path.abspath(ckpt_path), {
                'params': unreplicated_params,
                'opt_state': unreplicated_opt
            })
            tprint(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
