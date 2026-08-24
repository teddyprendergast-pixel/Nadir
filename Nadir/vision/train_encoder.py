import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Callable, Tuple

class SpatialSoftmaxFlax(nn.Module):
    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x_flat = x.reshape((b, -1, c))
        softmax_weights = jax.nn.softmax(x_flat, axis=1)
        softmax_weights = softmax_weights.reshape((b, h, w, c))
        
        # Coordinate grids
        pos_y, pos_x = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, h),
            jnp.linspace(-1.0, 1.0, w),
            indexing='ij'
        )
        
        expected_x = jnp.sum(softmax_weights * pos_x[..., None], axis=(1, 2))
        expected_y = jnp.sum(softmax_weights * pos_y[..., None], axis=(1, 2))
        
        coords = jnp.stack([expected_x, expected_y], axis=-1)  # (B, C, 2)
        return coords.reshape((b, c * 2))

class VisualEncoderFlax(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 1)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = SpatialSoftmaxFlax()(x)
        x = nn.Dense(features=32)(x)
        return x

def create_train_state(rng, learning_rate):
    encoder = VisualEncoderFlax()
    params = encoder.init(rng, jnp.ones([1, 64, 64, 1]))['params']
    tx = optax.adam(learning_rate)
    return encoder, params, tx

@jax.jit
def train_step(params, tx, opt_state, encoder, batch_depth, batch_target):
    def loss_fn(p):
        pred = encoder.apply({'params': p}, batch_depth)
        loss = jnp.mean((pred - batch_target) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

if __name__ == "__main__":
    print("Flax encoder training code initialized.")
    # Here you would load MJX data, run DAgger loop, etc.
