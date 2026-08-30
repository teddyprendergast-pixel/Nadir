"""PPO for the Nadir walking task, on top of the MJX environment.

The GAE recursion below distinguishes two masks, which the previous version
conflated into one:

* `terminated` — the robot actually fell. There is no future, so the value
  bootstrap is zeroed.
* `done` — the episode boundary, i.e. termination *or* the 20 s time limit.
  This resets the advantage trace so credit does not flow across episodes.

Zeroing the bootstrap on a time limit teaches the critic that the world ends at
20 s. Failing to reset the trace lets one episode's advantage bleed into the
next. Both matter; they are not the same mask.
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from .networks import ActorCritic


class Transition(NamedTuple):
    obs: jnp.ndarray
    privileged_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    terminated: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray


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
            optax.adam(learning_rate=config.learning_rate, eps=1e-5),
        )

    def compute_gae(self, transitions: Transition, last_value: jnp.ndarray):
        def _get_advantages(carry, transition):
            gae, next_value = carry
            done = transition.done.astype(jnp.float32)
            terminated = transition.terminated.astype(jnp.float32)

            delta = (
                transition.reward
                + self.config.gamma * next_value * (1.0 - terminated)
                - transition.value
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_value), last_value),
            transitions,
            reverse=True,
        )
        returns = advantages + transitions.value
        return advantages, returns

    def update(self, runner_state: RunnerState, transitions, advantages, returns):
        batch_size = self.config.num_steps * self.config.num_envs
        minibatch_size = batch_size // self.config.num_minibatches

        def _flatten(x):
            return x.reshape((batch_size,) + x.shape[2:])

        transitions_flat = jax.tree_util.tree_map(_flatten, transitions)
        advantages_flat = _flatten(advantages)
        returns_flat = _flatten(returns)

        if self.config.normalize_advantage:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                advantages_flat.std() + 1e-8
            )

        def _update_epoch(carry, _):
            params, opt_state, rng = carry
            rng, rng_perm = jax.random.split(rng)
            permutation = jax.random.permutation(rng_perm, batch_size)

            def _update_minibatch(carry, mb_idx):
                params, opt_state = carry
                mb_indices = jax.lax.dynamic_slice(
                    permutation, (mb_idx * minibatch_size,), (minibatch_size,)
                )

                mb_obs = transitions_flat.obs[mb_indices]
                mb_priv_obs = transitions_flat.privileged_obs[mb_indices]
                mb_actions = transitions_flat.action[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_old_log_prob = transitions_flat.log_prob[mb_indices]
                mb_old_values = transitions_flat.value[mb_indices]

                def _loss_fn(params):
                    values, log_prob, entropy = self.actor_critic.apply(
                        params, mb_obs, mb_priv_obs, mb_actions,
                        method=self.actor_critic.evaluate_action,
                    )

                    ratio = jnp.exp(log_prob - mb_old_log_prob)
                    surr1 = ratio * mb_advantages
                    surr2 = jnp.clip(
                        ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                    ) * mb_advantages
                    policy_loss = -jnp.minimum(surr1, surr2).mean()

                    v_loss_unclipped = jnp.square(values - mb_returns)
                    v_clipped = mb_old_values + jnp.clip(
                        values - mb_old_values, -self.config.clip_range, self.config.clip_range
                    )
                    v_loss_clipped = jnp.square(v_clipped - mb_returns)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()

                    total_loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy_loss
                    )

                    approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
                    clip_frac = jnp.mean(
                        (jnp.abs(ratio - 1.0) > self.config.clip_range).astype(jnp.float32)
                    )
                    return total_loss, (policy_loss, value_loss, entropy_loss, approx_kl, clip_frac)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss, aux), grads = grad_fn(params)

                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                return (params, opt_state), (loss,) + aux

            (params, opt_state), metrics = jax.lax.scan(
                _update_minibatch,
                (params, opt_state),
                jnp.arange(self.config.num_minibatches),
            )
            return (params, opt_state, rng), metrics

        (new_params, new_opt_state, new_rng), epoch_metrics = jax.lax.scan(
            _update_epoch,
            (runner_state.params, runner_state.opt_state, runner_state.rng),
            None,
            length=self.config.update_epochs,
        )

        new_runner_state = RunnerState(
            params=new_params,
            opt_state=new_opt_state,
            env_state=runner_state.env_state,
            obs=runner_state.obs,
            privileged_obs=runner_state.privileged_obs,
            rng=new_rng,
        )

        names = ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]
        avg_metrics = {n: m.mean() for n, m in zip(names, epoch_metrics)}
        return new_runner_state, avg_metrics

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, runner_state: RunnerState, _):
        def _env_step(carry, _):
            env_state, obs, priv_obs, rng = carry
            rng, rng_action = jax.random.split(rng)

            action, mean, log_std = self.actor_critic.apply(
                runner_state.params, obs, rng_action, method=self.actor_critic.get_action
            )
            value = self.actor_critic.apply(
                runner_state.params, priv_obs, method=self.actor_critic.get_value
            )

            std = jnp.exp(log_std)
            log_prob = -0.5 * (
                jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi)
            )
            log_prob = jnp.sum(log_prob, axis=-1)

            # The env owns its RNG streams, one per environment, so stepping
            # takes no key here. It hands back the completed episode's return
            # and length, which cannot be read off the returned state because
            # that state has already auto-reset.
            (next_env_state, next_obs, next_priv_obs, reward, done, terminated,
             episode_return, episode_length) = self.env.step(env_state, action)

            transition = Transition(
                obs=obs,
                privileged_obs=priv_obs,
                action=action,
                reward=reward,
                done=done,
                terminated=terminated,
                value=value,
                log_prob=log_prob,
                episode_return=episode_return,
                episode_length=episode_length,
            )
            return (next_env_state, next_obs, next_priv_obs, rng), transition

        (new_env_state, next_obs, next_priv_obs, new_rng), transitions = jax.lax.scan(
            _env_step,
            (runner_state.env_state, runner_state.obs, runner_state.privileged_obs, runner_state.rng),
            None,
            length=self.config.num_steps,
        )

        last_value = self.actor_critic.apply(
            runner_state.params, next_priv_obs, method=self.actor_critic.get_value
        )

        advantages, returns = self.compute_gae(transitions, last_value)

        state_for_update = RunnerState(
            params=runner_state.params,
            opt_state=runner_state.opt_state,
            env_state=new_env_state,
            obs=next_obs,
            privileged_obs=next_priv_obs,
            rng=new_rng,
        )

        final_runner_state, metrics = self.update(
            state_for_update, transitions, advantages, returns
        )

        # Episode statistics, averaged over the episodes that actually finished
        # in this rollout. `reward_sum` over a 24-step window says nothing about
        # whether the robot is walking; mean episode return and length do.
        done_f = transitions.done.astype(jnp.float32)
        n_done = done_f.sum()
        safe_n = jnp.maximum(n_done, 1.0)
        metrics["episode_return"] = (transitions.episode_return * done_f).sum() / safe_n
        metrics["episode_length"] = (transitions.episode_length * done_f).sum() / safe_n
        metrics["episodes_finished"] = n_done
        metrics["termination_rate"] = (
            transitions.terminated.astype(jnp.float32).sum() / safe_n
        )
        metrics["mean_step_reward"] = transitions.reward.mean()

        return final_runner_state, metrics
