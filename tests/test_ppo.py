"""PPO update-rule tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nadir.training.config import PPOConfig
from Nadir.training.networks import create_actor_critic
from Nadir.training.ppo import PPOTrainer, Transition


@pytest.fixture(scope="module")
def trainer():
    cfg = PPOConfig(num_envs=4, num_steps=8, num_minibatches=2, update_epochs=1)
    return PPOTrainer(cfg, env=None, actor_critic=create_actor_critic(cfg))


def _init_opt_state(trainer):
    params = trainer.actor_critic.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 47)), jnp.zeros((1, 51))
    )
    return trainer.optimizer.init(params)


def test_learning_rate_lives_in_the_optimiser_state(trainer):
    st = _init_opt_state(trainer)
    assert float(trainer._get_lr(st)) == pytest.approx(trainer.config.learning_rate)


def test_adaptive_lr_shrinks_when_kl_runs_hot(trainer):
    st = _init_opt_state(trainer)
    lr0 = float(trainer._get_lr(st))
    hot = trainer._adapt_lr(st, jnp.array(10.0 * trainer.config.target_kl))
    assert float(trainer._get_lr(hot)) < lr0


def test_adaptive_lr_grows_when_kl_has_headroom(trainer):
    st = _init_opt_state(trainer)
    lr0 = float(trainer._get_lr(st))
    cold = trainer._adapt_lr(st, jnp.array(0.1 * trainer.config.target_kl))
    assert float(trainer._get_lr(cold)) > lr0


def test_adaptive_lr_is_clamped(trainer):
    """The bounds hold, compared at the precision they are stored in.

    optax keeps the injected hyperparameter in float32, and float32(1e-5) is
    9.9999997e-06 — very slightly below the Python float 1e-5. Comparing
    across precisions here would fail on a controller that is behaving.
    """
    lo = float(np.float32(trainer.config.lr_min))
    hi = float(np.float32(trainer.config.lr_max))

    st = _init_opt_state(trainer)
    for _ in range(50):                      # sustained hot KL
        st = trainer._adapt_lr(st, jnp.array(1e3))
    assert float(trainer._get_lr(st)) >= lo
    assert float(trainer._get_lr(st)) == pytest.approx(lo, rel=1e-6)

    for _ in range(50):                      # sustained cold KL
        st = trainer._adapt_lr(st, jnp.array(0.0))
    assert float(trainer._get_lr(st)) <= hi
    assert float(trainer._get_lr(st)) == pytest.approx(hi, rel=1e-6)


def test_gae_zeroes_bootstrap_on_termination_but_not_on_truncation(trainer):
    """The distinction the shipped implementation collapsed into one mask."""
    n, e = trainer.config.num_steps, trainer.config.num_envs
    z = jnp.zeros((n, e))

    def make(done, terminated):
        return Transition(
            obs=z, privileged_obs=z, action=z,
            reward=jnp.ones((n, e)), done=done, terminated=terminated,
            value=z, log_prob=z, episode_return=z, episode_length=z,
        )

    last_value = jnp.full((e,), 100.0)
    ends = jnp.zeros((n, e)).at[-1].set(1.0)

    # Truncated at the last step: the future is still worth bootstrapping.
    adv_trunc, _ = trainer.compute_gae(make(ends, jnp.zeros((n, e))), last_value)
    # Terminated at the last step: there is no future.
    adv_term, _ = trainer.compute_gae(make(ends, ends), last_value)

    assert float(adv_trunc[-1].mean()) > float(adv_term[-1].mean()), (
        "truncation must not zero the value bootstrap the way termination does"
    )
