import os
import yaml
from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class DomainRandParams:
    mass_multiplier: jnp.ndarray
    com_offset: jnp.ndarray
    ground_friction: jnp.ndarray
    actuator_kp_multiplier: jnp.ndarray
    actuator_kv_multiplier: jnp.ndarray
    control_latency: jnp.ndarray
    push_impulse: jnp.ndarray
    compliance_factor: jnp.ndarray

def load_hardware_config():
    """Reads actuator measured ranges if available."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "hardware", "measured", "actuators.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def randomize_domain(rng: jnp.ndarray, num_envs: int, is_forest: bool = False) -> DomainRandParams:
    """Generate domain randomization parameters for all envs."""
    rngs = jax.random.split(rng, 8)
    
    mass_mult = jax.random.uniform(rngs[0], shape=(num_envs,), minval=0.85, maxval=1.15)
    com_offset = jax.random.uniform(rngs[1], shape=(num_envs, 3), minval=-0.01, maxval=0.01)
    
    if is_forest:
        friction_min, friction_max = 0.3, 0.9 # Wet surface
        compliance = jax.random.uniform(rngs[7], shape=(num_envs,), minval=0.5, maxval=1.0)
    else:
        friction_min, friction_max = 0.4, 1.2
        compliance = jnp.ones(num_envs)
        
    friction = jax.random.uniform(rngs[2], shape=(num_envs,), minval=friction_min, maxval=friction_max)
    kp_mult = jax.random.uniform(rngs[3], shape=(num_envs,), minval=0.8, maxval=1.2)
    kv_mult = jax.random.uniform(rngs[4], shape=(num_envs,), minval=0.8, maxval=1.2)
    
    # 0 to 2 steps latency
    latency = jax.random.randint(rngs[5], shape=(num_envs,), minval=0, maxval=3)
    
    # Random push impulse: 0.5 - 2.0 N
    push_mag = jax.random.uniform(rngs[6], shape=(num_envs, 3), minval=-2.0, maxval=2.0)
    # Mask out some pushes to make them sporadic
    push_mask = jax.random.bernoulli(rngs[7], p=0.05, shape=(num_envs, 1))
    push_impulse = push_mag * push_mask
    
    return DomainRandParams(
        mass_multiplier=mass_mult,
        com_offset=com_offset,
        ground_friction=friction,
        actuator_kp_multiplier=kp_mult,
        actuator_kv_multiplier=kv_mult,
        control_latency=latency,
        push_impulse=push_impulse,
        compliance_factor=compliance
    )
