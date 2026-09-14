import os
import sys

# Auto-relaunch using the project's virtual environment if invoked with a different Python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
venv_python = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

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
import re
from datetime import datetime, timedelta

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

# Ensure repository root is in sys.path for direct script execution
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from .config import PPOConfig, EnvConfig
    from .networks import create_actor_critic
    from .ppo_cpu import PPOTrainerCPU, RunnerState
except ImportError:
    from nadir.training.config import PPOConfig, EnvConfig
    from nadir.training.networks import create_actor_critic
    from nadir.training.ppo_cpu import PPOTrainerCPU, RunnerState

from nadir.sim.env_mjx import NadirEnv

def format_duration(seconds: float) -> str:
    secs = int(max(0, seconds))
    days, secs = divmod(secs, 86400)
    hours, secs = divmod(secs, 3600)
    mins, secs = divmod(secs, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts) if parts else "0s"

def parse_steps_or_time(val: str, estimated_sps: float = 220.0) -> tuple[int, datetime | None]:
    """Parse user input for steps (e.g. '10M', '500k', '2000000') or target time (e.g. '8pm', '20:00').
    Returns (steps, target_time_if_any).
    """
    val = val.strip()
    # Check if number with optional k/m suffix
    num_match = re.match(r'^([\d\.]+)\s*([kKmM])?$', val)
    if num_match:
        num = float(num_match.group(1))
        if num <= 0:
            raise ValueError("Steps must be greater than 0")
        suffix = (num_match.group(2) or '').upper()
        if suffix == 'K':
            num *= 1_000
        elif suffix == 'M':
            num *= 1_000_000
        steps = int(num)
        return steps, None

    # Check if time format e.g. '8pm', '20:00', 'until 8pm'
    time_str = val
    if time_str.lower().startswith('until '):
        time_str = time_str[6:].strip()
    now = datetime.now()
    for fmt in ('%I:%M:%S%p', '%I:%M%p', '%I%p', '%I:%M:%S %p', '%I:%M %p', '%I %p', '%H:%M:%S', '%H:%M'):
        try:
            clean_str = re.sub(r'\s+', ' ', time_str)
            parsed_time = datetime.strptime(clean_str, fmt).time()
            target_dt = datetime.combine(now.date(), parsed_time)
            if target_dt <= now:
                target_dt += timedelta(days=1)
            diff_secs = (target_dt - now).total_seconds()
            steps = max(1, int(diff_secs * estimated_sps))
            return steps, target_dt
        except ValueError:
            pass

    raise ValueError(f"Unrecognized format '{val}'. Enter e.g. '1M', '500k', '2000000', or '8pm'")

def interactive_step_prompt(default_steps: int = 10_000_000, estimated_sps: float = 220.0) -> tuple[int, datetime | None]:
    current_steps = default_steps
    while True:
        prompt_str = f"\nHow many steps do you want to run? [e.g. 500k, 1M, 10M, or '8pm'] (default: {current_steps:,}): "
        try:
            val = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if not val:
            steps = current_steps
            target_time = None
        else:
            try:
                steps, target_time = parse_steps_or_time(val, estimated_sps)
            except Exception as e:
                print(f"Invalid input: {e}. Please enter a valid number of steps or a target time.")
                continue

        est_seconds = steps / estimated_sps
        finish_dt = target_time if target_time else (datetime.now() + timedelta(seconds=est_seconds))
        dur_str = format_duration(est_seconds)
        finish_day = "today" if finish_dt.date() == datetime.now().date() else ("tomorrow" if finish_dt.date() == (datetime.now() + timedelta(days=1)).date() else finish_dt.strftime('%Y-%m-%d'))

        print("\n" + "=" * 60)
        print(" TRAINING ESTIMATION")
        print("=" * 60)
        print(f" Requested Steps:    {steps:,}")
        print(f" Estimated Speed:    ~{estimated_sps:.0f} steps/second (CPU)")
        print(f" Estimated Duration: {dur_str}")
        print(f" Estimated Finish:   {finish_dt.strftime('%I:%M:%S %p')} ({finish_day})")
        print("=" * 60)

        try:
            confirm = input("Are you happy with this? [y/n/q] (default: y): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if confirm in ("y", "yes", ""):
            print(f"\nStarting simulation with {steps:,} steps...\n")
            return steps, target_time
        elif confirm in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        else:
            print("\nOkay, let's adjust the number of steps.")
            current_steps = steps

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
    parser.add_argument("--until", type=str, default=None, help="Time to run training until (e.g. '8pm', '20:00')")
    parser.add_argument("--estimated-sps", type=float, default=220.0, help="Estimated steps per second for duration calculation (default: 220)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip interactive confirmation prompt")
    parser.add_argument("--no-interactive", action="store_true", help="Run non-interactively without prompt")
    args = parser.parse_args()

    target_time = None
    total_timesteps = args.total_timesteps
    if args.until:
        _, target_time = parse_steps_or_time(args.until, args.estimated_sps)

    # Interactive step prompt and estimation
    if not args.yes and not args.no_interactive and sys.stdin.isatty():
        default_prompt_steps = total_timesteps if total_timesteps != 100_000_000 else 10_000_000
        total_timesteps, target_time = interactive_step_prompt(
            default_steps=default_prompt_steps,
            estimated_sps=args.estimated_sps
        )

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
        total_timesteps=total_timesteps,
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
    num_updates = max(1, ppo_config.total_timesteps // (ppo_config.num_steps * ppo_config.num_envs))
    print(f"Starting CPU-parallel training for {num_updates} updates across {active_devices} CPU devices...")
    
    # Compile pmap train step
    pmap_train_step = jax.pmap(trainer.train_step, axis_name="devices")
    
    last_saved_update = 0
    last_completed_update = 0

    def save_checkpoint(upd: int, suffix: str = ""):
        nonlocal last_saved_update
        name = f"update_{upd}{suffix}"
        ckpt_path = os.path.join(ckpt_dir, name)
        unreplicated_params = flax_utils.unreplicate(runner_state.params)
        unreplicated_opt = flax_utils.unreplicate(runner_state.opt_state)
        checkpointer.save(os.path.abspath(ckpt_path), {
            'params': unreplicated_params,
            'opt_state': unreplicated_opt
        })
        last_saved_update = upd
        tprint(f"Saved checkpoint to {ckpt_path}")

    try:
        for update in tqdm(range(1, num_updates + 1), desc="Training CPU", unit="update"):
            update_start_time = time.time()
            runner_state, metrics = pmap_train_step(runner_state, None)
            
            # Block until computation is complete
            metrics['reward_sum'].block_until_ready()
            update_time = time.time() - update_start_time
            last_completed_update = update
            
            if update % ppo_config.log_interval == 0:
                sps = (ppo_config.num_steps * ppo_config.num_envs) / update_time if update_time > 0 else 0
                tprint(f"Update: {update}/{num_updates}")
                tprint(f"Reward Sum: {float(metrics['reward_sum'][0]):.2f}")
                tprint(f"Policy Loss: {float(metrics['policy_loss'][0]):.4f}")
                tprint(f"Value Loss: {float(metrics['value_loss'][0]):.4f}")
                tprint(f"SPS: {sps:.0f}")
                if target_time:
                    remaining = target_time - datetime.now()
                    rem_secs = max(0, int(remaining.total_seconds()))
                    tprint(f"Time Remaining: {format_duration(rem_secs)} (until {target_time.strftime('%I:%M:%S %p')})")
                tprint("-" * 30)
                
            if update % ppo_config.save_interval == 0:
                save_checkpoint(update)

            if target_time and datetime.now() >= target_time:
                tprint(f"\n[INFO] Target time reached ({target_time.strftime('%I:%M:%S %p')}). Stopping training after update {update}.")
                break
    except KeyboardInterrupt:
        tprint(f"\nTraining interrupted by user at update {last_completed_update}.")
    finally:
        if last_completed_update > 0 and last_saved_update != last_completed_update:
            tprint("Saving final checkpoint before exit...")
            save_checkpoint(last_completed_update, suffix="_final")
        try:
            checkpointer.wait_until_finished()
            checkpointer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
