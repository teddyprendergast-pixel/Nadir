import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import NamedTuple, Any, Tuple, Dict
from functools import partial
from .networks import ActorCritic

class Transition(NamedTuple):
    obs: jnp.ndarray
    privileged_obs: jnp.ndarray  
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray

class RunnerState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    obs: jnp.ndarray
    privileged_obs: jnp.ndarray
    rng: jax.Array

class PPOTrainer:
    def __init__(self, config, env, actor_critic: ActorCritic):
        self.config = config
        self.env = env
        self.actor_critic = actor_critic
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=config.learning_rate, eps=1e-5)
        )
    
    def compute_gae(self, transitions: Transition, last_value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generalized Advantage Estimation."""
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            
            delta = reward + self.config.gamma * next_value * (1 - done) - value
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_value), last_value),
            transitions,
            reverse=True,
        )
        returns = advantages + transitions.value
        return advantages, returns
    
    def update(self, runner_state: RunnerState, transitions: Transition, advantages: jnp.ndarray, returns: jnp.ndarray) -> Tuple[RunnerState, Dict[str, jnp.ndarray]]:
        """PPO clipped objective update."""
        
        # Flatten the batch dimensions (num_steps, num_envs) -> (num_steps * num_envs)
        batch_size = self.config.num_steps * self.config.num_envs
        minibatch_size = batch_size // self.config.num_minibatches

        def _flatten(x):
            return x.reshape((batch_size,) + x.shape[2:])

        transitions_flat = jax.tree_util.tree_map(_flatten, transitions)
        advantages_flat = _flatten(advantages)
        returns_flat = _flatten(returns)

        if self.config.normalize_advantage:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
            
        def _update_epoch(carry, _):
            params, opt_state, rng = carry
            
            rng, rng_perm = jax.random.split(rng)
            permutation = jax.random.permutation(rng_perm, batch_size)
            
            def _update_minibatch(carry, mb_idx):
                params, opt_state = carry
                
                # Fetch minibatch
                mb_indices = jax.lax.dynamic_slice(permutation, (mb_idx * minibatch_size,), (minibatch_size,))
                
                mb_obs = transitions_flat.obs[mb_indices]
                mb_priv_obs = transitions_flat.privileged_obs[mb_indices]
                mb_actions = transitions_flat.action[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_old_log_prob = transitions_flat.log_prob[mb_indices]
                mb_old_values = transitions_flat.value[mb_indices]

                def _loss_fn(params):
                    values, log_prob, entropy = self.actor_critic.apply(
                        params, mb_obs, mb_priv_obs, mb_actions, method=self.actor_critic.evaluate_action
                    )
                    
                    # Policy loss
                    ratio = jnp.exp(log_prob - mb_old_log_prob)
                    surr1 = ratio * mb_advantages
                    surr2 = jnp.clip(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * mb_advantages
                    policy_loss = -jnp.minimum(surr1, surr2).mean()
                    
                    # Value loss
                    v_loss_unclipped = jnp.square(values - mb_returns)
                    v_clipped = mb_old_values + jnp.clip(values - mb_old_values, -self.config.clip_range, self.config.clip_range)
                    v_loss_clipped = jnp.square(v_clipped - mb_returns)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
                    
                    # Entropy
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
                    
                    return total_loss, (policy_loss, value_loss, entropy_loss)
                
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(params)
                
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                return (params, opt_state), (loss, policy_loss, value_loss, entropy_loss)

            (params, opt_state), metrics = jax.lax.scan(
                _update_minibatch,
                (params, opt_state),
                jnp.arange(self.config.num_minibatches)
            )
            
            return (params, opt_state, rng), metrics

        (new_params, new_opt_state, new_rng), epoch_metrics = jax.lax.scan(
            _update_epoch,
            (runner_state.params, runner_state.opt_state, runner_state.rng),
            None,
            length=self.config.update_epochs
        )
        
        new_runner_state = RunnerState(
            params=new_params,
            opt_state=new_opt_state,
            env_state=runner_state.env_state,
            obs=runner_state.obs,
            privileged_obs=runner_state.privileged_obs,
            rng=new_rng
        )
        
        avg_metrics = {
            "loss": epoch_metrics[0].mean(),
            "policy_loss": epoch_metrics[1].mean(),
            "value_loss": epoch_metrics[2].mean(),
            "entropy_loss": epoch_metrics[3].mean()
        }
        
        return new_runner_state, avg_metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, runner_state: RunnerState, _) -> Tuple[RunnerState, Dict[str, jnp.ndarray]]:
        """Single train step: collect rollout + update."""
        
        def _env_step(carry, _):
            env_state, obs, priv_obs, rng = carry
            
            rng, rng_action = jax.random.split(rng)
            
            # Get action
            action, mean, log_std = self.actor_critic.apply(
                runner_state.params, obs, rng_action, method=self.actor_critic.get_action
            )
            
            # Get value
            value = self.actor_critic.apply(
                runner_state.params, priv_obs, method=self.actor_critic.get_value
            )
            
            # Get log prob
            std = jnp.exp(log_std)
            log_prob = -0.5 * (jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi))
            log_prob = jnp.sum(log_prob, axis=-1)
            
            # Step env
            rng, rng_env = jax.random.split(rng)
            next_env_state, next_obs, next_priv_obs, reward, done = self.env.step(rng_env, env_state, action)
            
            transition = Transition(
                obs=obs,
                privileged_obs=priv_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob
            )
            
            return (next_env_state, next_obs, next_priv_obs, rng), transition

        # Collect rollout
        (new_env_state, next_obs, next_priv_obs, new_rng), transitions = jax.lax.scan(
            _env_step,
            (runner_state.env_state, runner_state.obs, runner_state.privileged_obs, runner_state.rng),
            None,
            length=self.config.num_steps
        )
        
        # Compute last value
        last_value = self.actor_critic.apply(
            runner_state.params, next_priv_obs, method=self.actor_critic.get_value
        )
        
        # Compute GAE
        advantages, returns = self.compute_gae(transitions, last_value)
        
        # Prepare runner state for update
        state_for_update = RunnerState(
            params=runner_state.params,
            opt_state=runner_state.opt_state,
            env_state=new_env_state,
            obs=next_obs,
            privileged_obs=next_priv_obs,
            rng=new_rng
        )
        
        # Update
        final_runner_state, metrics = self.update(state_for_update, transitions, advantages, returns)
        
        metrics["reward_sum"] = transitions.reward.sum(axis=0).mean()
        metrics["episode_lengths"] = jnp.logical_not(transitions.done).sum(axis=0).mean()
        
        return final_runner_state, metrics
