"""
===============================================================================
Nadir Robot — AI Brain Exporter (JAX/Flax to ONNX)
===============================================================================
What this file does:
  This script takes the trained AI neural network "brain" from the training
  simulator and packages it into a compact, standardized file format called ONNX
  (e.g., `nadir_policy.onnx`).

Why this matters for non-coders / beginners:
  - JAX and Python are wonderful for training because they can simulate 4,000
    robots at once on big computers.
  - BUT when it is time to put the brain inside the *real* physical robot
    (which runs on a small, low-power computer like a Raspberry Pi or ARM board),
    we need the brain to run as fast as possible (< 2 milliseconds per thought!).
  - "ONNX" (Open Neural Network Exchange) is a universal standard for neural networks.
    Once exported to ONNX, the brain can run in C++, Python, or embedded Linux
    without needing heavy training libraries!
===============================================================================
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple


def export_actor_to_onnx(params, actor_network, obs_dim: int = 41,
                          output_path: str = 'nadir_policy.onnx'):
    """
    Exports the trained Actor (walking policy) to an ONNX file.

    How it works under the hood:
      1. JAX/Flax stores neural network weights (numbers that represent knowledge)
         in a nested Python structure.
      2. We extract these weights layer-by-layer and copy them into an equivalent
         PyTorch neural network architecture.
      3. We tell PyTorch to export the graph into the universal `.onnx` format.
      This elegant bridge avoids complex compiler dependencies like jax2tf!
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for exporting the brain to ONNX. "
            "Please install it using: pip install torch"
        )
        
    # Define a simple PyTorch model that matches the layers of our Flax Actor
    class PyTorchActor(nn.Module):
        def __init__(self, flax_params):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Extract weights and biases from the Flax parameter tree
            flat_params = jax.tree_util.tree_leaves_with_path(flax_params)
            
            layers_dict = {}
            for path, param in flat_params:
                # Path typically looks like: params -> Dense_0 -> kernel/bias
                layer_name = path[1].key
                param_type = path[2].key
                if layer_name not in layers_dict:
                    layers_dict[layer_name] = {}
                layers_dict[layer_name][param_type] = np.array(param)
                
            # Sort layers in sequential order (Dense_0, Dense_1, Dense_2, etc.)
            sorted_layers = sorted(layers_dict.items(), key=lambda x: x[0])
            for name, weights in sorted_layers:
                w = weights['kernel']
                b = weights['bias']
                # Create a matching Linear (Dense) layer in PyTorch
                linear = nn.Linear(w.shape[0], w.shape[1])
                # Copy the trained weights into PyTorch (transposing because PyTorch expects [out, in])
                linear.weight.data = torch.tensor(w.T, dtype=torch.float32)
                linear.bias.data = torch.tensor(b, dtype=torch.float32)
                self.layers.append(linear)
                
        def forward(self, x):
            # Pass sensory observation through each layer with ReLU activation
            for i, layer in enumerate(self.layers):
                x = layer(x)
                # Apply ReLU activation between hidden layers (but not on the final output layer)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
            return x
            
    # Build the PyTorch model and put it into evaluation (read-only) mode
    torch_model = PyTorchActor(params)
    torch_model.eval()
    
    # Create a dummy sensory input (41 numbers) so PyTorch can trace the math operations
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)
    
    # Export the model graph to the destination .onnx file
    torch.onnx.export(
        torch_model,
        dummy_input,
        output_path,
        export_params=True,             # Embed the learned weights directly inside the file
        opset_version=14,               # Standard modern ONNX operator set version
        do_constant_folding=True,       # Optimize and pre-calculate constant math
        input_names=['observation'],    # Input sensory data name (41 values)
        output_names=['action'],        # Output motor angle command name (10 values)
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
    )
    print(f"Successfully exported ONNX brain to: {output_path}")


def validate_export(params, actor_network, onnx_path: str, obs_dim: int = 41, num_tests: int = 100):
    """
    Safety Check: Validates that the exported ONNX brain produces the EXACT SAME
    decisions as the original JAX simulation brain.

    Why this matters:
      Before putting an exported brain onto a real walking robot, we test 100
      random situations on both models. If their motor output numbers match to within
      0.001, we know the export was 100% perfect and safe!
    """
    import onnxruntime as ort
    
    # Load the exported ONNX model
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    rng = jax.random.PRNGKey(0)
    for test_idx in range(num_tests):
        rng, key = jax.random.split(rng)
        # Generate random test senses (balance, speeds, angles)
        test_obs = jax.random.normal(key, (1, obs_dim))
        
        # 1. Ask original JAX model for its decision
        jax_out = actor_network.apply(params, test_obs)
        
        # 2. Ask exported ONNX model for its decision
        onnx_out = sess.run(None, {input_name: np.array(test_obs)})[0]
        
        # 3. Compare their outputs: assert that they are nearly identical
        np.testing.assert_allclose(jax_out, onnx_out, rtol=1e-3, atol=1e-5)
        
    print(f"ONNX model validated successfully against JAX original ({num_tests} test cases passed).")
