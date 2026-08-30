"""Export the trained Flax actor to ONNX for the Pi.

Built directly with the `onnx` helper API — no PyTorch, no TensorFlow. The Pi 4
has ~1 GB of RAM and runs ONNX Runtime; pulling a training framework into the
export path to convert four Dense layers is not a trade worth making.

The previous implementation had three defects that would each have produced a
policy that silently did not match the trained one:

* it applied `torch.relu` between layers while the network trains with ELU;
* it flattened the parameter tree assuming `params/Dense_k/...`, but the
  ActorCritic nests submodules as `params/actor/Dense_k/...`, so it mixed
  critic weights into the actor;
* it ordered layers by string sort, which breaks at Dense_10.

The exported graph emits both the raw action and the joint goal positions in
radians, so the deploy side does not have to restate DEFAULT_POSE and
ACTION_SCALE and cannot drift out of sync with training.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..sim.env_mjx import NadirEnv


def _actor_layers(params):
    """Pull the actor's Dense layers out of the Flax parameter tree, in order."""
    tree = params["params"] if "params" in params else params
    actor = tree["actor"]
    names = sorted(
        (k for k in actor if k.startswith("Dense_")),
        key=lambda k: int(k.split("_")[1]),   # numeric, not lexicographic
    )
    return [(np.asarray(actor[n]["kernel"]), np.asarray(actor[n]["bias"])) for n in names]


def build_onnx(params, obs_dim, default_pose, action_scale, ctrl_lower, ctrl_upper,
               activation="elu"):
    layers = _actor_layers(params)
    if not layers:
        raise ValueError("no Dense layers found under params/actor")

    inits, nodes = [], []

    def const(name, arr):
        t = numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)
        inits.append(t)
        return name

    x = "observation"
    for i, (w, b) in enumerate(layers):
        wn, bn = const(f"W{i}", w), const(f"b{i}", b)
        y = f"gemm{i}"
        nodes.append(helper.make_node("Gemm", [x, wn, bn], [y], name=f"dense_{i}"))
        x = y
        if i < len(layers) - 1:                       # no activation on the output
            act = "Elu" if activation == "elu" else "Relu"
            y = f"act{i}"
            nodes.append(helper.make_node(act, [x], [y], name=f"{act.lower()}_{i}"))
            x = y

    # Deterministic action: the distribution mean, clipped to the range the
    # policy was trained to emit. No sampling at deploy time.
    # ONNX Clip takes *scalar* min/max, so the symmetric [-1, 1] bound uses
    # Clip while the per-joint ctrlrange below needs broadcasting Max/Min.
    nodes.append(helper.make_node(
        "Clip", [x, const("a_lo", np.float32(-1.0)), const("a_hi", np.float32(1.0))],
        ["action"], name="clip_action"))

    # goal = clip(default_pose + action * action_scale, ctrlrange), per joint
    nodes.append(helper.make_node(
        "Mul", ["action", const("action_scale", action_scale)], ["scaled"], name="scale"))
    nodes.append(helper.make_node(
        "Add", ["scaled", const("default_pose", default_pose)], ["goal_raw"], name="offset"))
    nodes.append(helper.make_node(
        "Max", ["goal_raw", const("ctrl_lo", ctrl_lower)], ["goal_lo"], name="clamp_lo"))
    nodes.append(helper.make_node(
        "Min", ["goal_lo", const("ctrl_hi", ctrl_upper)], ["goal_position"], name="clamp_hi"))

    nu = len(default_pose)
    graph = helper.make_graph(
        nodes, "nadir_actor",
        inputs=[helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", obs_dim])],
        outputs=[
            helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", nu]),
            helper.make_tensor_value_info("goal_position", TensorProto.FLOAT, ["batch", nu]),
        ],
        initializer=inits,
    )
    model = helper.make_model(
        graph, producer_name="nadir",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    onnx.checker.check_model(model)
    return model


def validate(model_path, params, actor_critic, obs_dim, num_tests=256, tol=1e-5):
    """Check the ONNX graph reproduces the JAX actor's mean action.

    The JAX side is forced to full float32. On an A100, JAX defaults to TF32
    for matmuls, which carries ~10 mantissa bits and disagrees with ONNX
    Runtime's float32 CPU kernels at the 1e-3 level. That failure looks
    exactly like a broken export but is the *reference* being imprecise, not
    the graph — so pin the precision rather than loosening the tolerance.
    """
    import jax
    import jax.numpy as jnp
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1   # also silences pthread_setaffinity spam on HPC nodes
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (num_tests, obs_dim)) * 0.5

    with jax.default_matmul_precision("highest"):
        mean, _ = actor_critic.apply(params, obs, method=lambda m, o: m.actor(o))
    jax_action = np.asarray(jnp.clip(mean, -1.0, 1.0))

    onnx_action = sess.run(["action"], {in_name: np.asarray(obs, dtype=np.float32)})[0]

    max_err = float(np.max(np.abs(jax_action - onnx_action)))
    if max_err > tol:
        raise AssertionError(f"ONNX/JAX mismatch: max abs error {max_err:.3e} > {tol:.1e}")
    print(f"ONNX validated against JAX over {num_tests} observations "
          f"(max abs error {max_err:.2e})")
    return max_err


def main():
    p = argparse.ArgumentParser(description="Export the Nadir actor to ONNX")
    p.add_argument("--checkpoint", required=True, help="orbax checkpoint dir (e.g. .../best)")
    p.add_argument("--output", default="exported_models/nadir_policy.onnx")
    p.add_argument("--obs-dim", type=int, default=47)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    import orbax.checkpoint as ocp
    from ..training.config import PPOConfig
    from ..training.networks import create_actor_critic

    ckpt = ocp.StandardCheckpointer().restore(os.path.abspath(args.checkpoint))
    params = ckpt["params"]

    env = NadirEnv(num_envs=1)
    model = build_onnx(
        params,
        obs_dim=args.obs_dim,
        default_pose=np.array(env.DEFAULT_POSE),
        action_scale=np.array(env.ACTION_SCALE),
        ctrl_lower=np.asarray(env.ctrl_lower),
        ctrl_upper=np.asarray(env.ctrl_upper),
        activation=PPOConfig.activation,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save(model, args.output)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"Wrote {args.output} ({size_kb:.1f} KB)")

    if not args.no_validate:
        validate(args.output, params, create_actor_critic(PPOConfig()), args.obs_dim)


if __name__ == "__main__":
    main()
