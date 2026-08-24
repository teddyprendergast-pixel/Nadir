import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple

def export_actor_to_onnx(params, actor_network, obs_dim: int = 41,
                          output_path: str = 'nadir_policy.onnx'):
    """Export trained Flax actor network to ONNX format.
    
    Strategy: Extract weights from Flax params -> build equivalent PyTorch model -> export ONNX.
    This avoids jax2tf dependency issues.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for ONNX export (flax -> pytorch -> onnx strategy)")
        
    class PyTorchActor(nn.Module):
        def __init__(self, flax_params):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Assuming simple MLP structure from Flax Dense layers
            # Structure depends on the precise model in actor_network.
            # E.g. params = {'params': {'Dense_0': {'kernel': ..., 'bias': ...}, ...}}
            flat_params = jax.tree_util.tree_leaves_with_path(flax_params)
            
            # Parse parameters into linear layers
            # This is a simplified extraction; adjust based on actual Flax architecture
            layers_dict = {}
            for path, param in flat_params:
                # Path typically looks like ((DictKey(key='params'), DictKey(key='Dense_0'), DictKey(key='kernel')), ...)
                layer_name = path[1].key
                param_type = path[2].key
                if layer_name not in layers_dict:
                    layers_dict[layer_name] = {}
                layers_dict[layer_name][param_type] = np.array(param)
                
            sorted_layers = sorted(layers_dict.items(), key=lambda x: x[0])
            for name, weights in sorted_layers:
                w = weights['kernel']
                b = weights['bias']
                linear = nn.Linear(w.shape[0], w.shape[1])
                linear.weight.data = torch.tensor(w.T, dtype=torch.float32)
                linear.bias.data = torch.tensor(b, dtype=torch.float32)
                self.layers.append(linear)
                
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)  # Adjust activation if necessary (e.g. elu)
            return x
            
    # Create and export PyTorch model
    torch_model = PyTorchActor(params)
    torch_model.eval()
    
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        torch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {output_path}")

def validate_export(params, actor_network, onnx_path: str, obs_dim: int = 41, num_tests: int = 100):
    """Validate ONNX model produces identical outputs to JAX model."""
    import onnxruntime as ort
    
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    rng = jax.random.PRNGKey(0)
    for _ in range(num_tests):
        rng, key = jax.random.split(rng)
        test_obs = jax.random.normal(key, (1, obs_dim))
        
        # JAX output
        jax_out = actor_network.apply(params, test_obs)
        
        # ONNX output
        onnx_out = sess.run(None, {input_name: np.array(test_obs)})[0]
        
        # Compare
        np.testing.assert_allclose(jax_out, onnx_out, rtol=1e-3, atol=1e-5)
        
    print("ONNX model validated successfully against JAX original.")
