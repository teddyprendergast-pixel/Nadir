import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Tuple

class ActorNetwork(nn.Module):
    """Gaussian policy network for PPO."""
    hidden_dims: Sequence[int] = (256, 256, 128)
    action_dim: int = 10
    activation: Callable = nn.elu
    init_noise_std: float = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = self.activation(x)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        # Learnable log_std (not state-dependent)
        log_std = self.param('log_std', nn.initializers.constant(jnp.log(self.init_noise_std)), (self.action_dim,))
        return mean, jnp.broadcast_to(log_std, mean.shape)

class CriticNetwork(nn.Module):
    """Value function network."""
    hidden_dims: Sequence[int] = (256, 256, 128)
    activation: Callable = nn.elu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = self.activation(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)

class ActorCritic(nn.Module):
    """Combined actor-critic with asymmetric observations."""
    actor_hidden_dims: Sequence[int]
    critic_hidden_dims: Sequence[int]
    action_dim: int
    activation_str: str = 'elu'
    init_noise_std: float = 1.0
    
    def setup(self):
        activation_fn = nn.elu if self.activation_str == 'elu' else nn.relu
        self.actor = ActorNetwork(
            hidden_dims=self.actor_hidden_dims,
            action_dim=self.action_dim,
            activation=activation_fn,
            init_noise_std=self.init_noise_std
        )
        self.critic = CriticNetwork(
            hidden_dims=self.critic_hidden_dims,
            activation=activation_fn
        )
    
    def __call__(self, obs: jnp.ndarray, privileged_obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor(obs)
        value = self.critic(privileged_obs)
        return mean, log_std, value
    
    def get_value(self, privileged_obs: jnp.ndarray) -> jnp.ndarray:
        return self.critic(privileged_obs)
    
    def get_action(self, obs: jnp.ndarray, rng: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor(obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, shape=mean.shape)
        action = mean + std * noise
        return action, mean, log_std
    
    def evaluate_action(self, obs: jnp.ndarray, privileged_obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor(obs)
        std = jnp.exp(log_std)
        value = self.critic(privileged_obs)
        # Gaussian log probability
        log_prob = -0.5 * (jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=-1)
        # Entropy
        entropy = 0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        return value, log_prob, entropy

def create_actor_critic(config) -> ActorCritic:
    """Helper to instantiate the ActorCritic network from config."""
    return ActorCritic(
        actor_hidden_dims=config.actor_hidden_dims,
        critic_hidden_dims=config.critic_hidden_dims,
        action_dim=10,
        activation_str=config.activation,
        init_noise_std=config.init_noise_std
    )
