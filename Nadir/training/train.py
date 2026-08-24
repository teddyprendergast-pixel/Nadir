import os
import argparse
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from datetime import datetime

from .config import PPOConfig, EnvConfig
from .networks import create_actor_critic
from .ppo import PPOTrainer, RunnerState
from nadir.sim.env_mjx import NadirEnv

def main():
    parser = argparse.ArgumentParser(description="Train Nadir bipedal robot policy")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=24, help="Number of steps per rollout")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup configs
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
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
    for update in range(1, num_updates + 1):
        runner_state, metrics = trainer.train_step(runner_state, None)
        
        if update % ppo_config.log_interval == 0:
            print(f"Update: {update}/{num_updates}")
            print(f"Reward Sum: {metrics['reward_sum']:.2f}")
            print(f"Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"Value Loss: {metrics['value_loss']:.4f}")
            print("-" * 30)
            
        if update % ppo_config.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"update_{update}")
            checkpointer.save(os.path.abspath(ckpt_path), {
                'params': runner_state.params,
                'opt_state': runner_state.opt_state
            })
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
