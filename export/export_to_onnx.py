"""Export a trained Neural Cellular Automata Keras model to ONNX.

The notebook trains an `update_model` that takes a single (3, 3, C) patch
and returns a (C,) update delta. For browser inference we rewrap the weights
as a fully convolutional model that operates on the whole (H, W, C) grid in
one pass:

    grid -> Conv2D(3x3, same)+ReLU -> Conv2D(1x1)  -> update delta

The 1x1 Conv2D is mathematically equivalent to the original Flatten+Dense,
because after the first conv (with same padding) every spatial position holds
a 64-dim feature vector, and Dense(64 -> C) applied per position is just a
1x1 convolution with the same weights.

This convolutional form is far cheaper to run, exports cleanly to ONNX, and
uses only ops (Conv, Relu) that are supported by the onnxruntime-web WebGL
backend.

Usage:
    python export/export_to_onnx.py \\
        --weights weights/phase4_run4 \\
        --out web/nca.onnx \\
        --height 64 --width 96
"""
import argparse
import os

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras import layers, models


def build_grid_model(channels: int, hidden: int, height, width):
    """Build a fully convolutional NCA update network.

    `height` / `width` may be None to keep the model fully dynamic; some
    onnxruntime-web backends prefer fixed shapes, so the CLI defaults to
    concrete values.
    """
    inputs = layers.Input(shape=(height, width, channels), name="grid")
    x = layers.Conv2D(
        hidden, kernel_size=(3, 3), padding="same",
        activation="relu", name="perceive",
    )(inputs)
    x = layers.Conv2D(
        channels, kernel_size=(1, 1), padding="same",
        kernel_initializer="zeros", bias_initializer="zeros",
        name="update",
    )(x)
    return models.Model(inputs, x, name="nca_update")


def copy_weights(src_patch_model: tf.keras.Model, dst_grid_model: tf.keras.Model):
    """Copy weights from the patch-based training model to the grid model.

    Source layers (training notebook):
        conv2d_1: Conv2D(hidden, 3x3, valid) — kernel (3, 3, C, H), bias (H,)
        dense:    Dense(C)                   — kernel (H, C),       bias (C,)

    Destination:
        perceive: Conv2D(hidden, 3x3, same)  — same kernel/bias as conv2d_1
        update:   Conv2D(C, 1x1)             — kernel (1, 1, H, C),  bias (C,)
                                               weights = dense reshaped
    """
    src_conv = None
    src_dense = None
    for layer in src_patch_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and src_conv is None:
            src_conv = layer
        elif isinstance(layer, tf.keras.layers.Dense):
            src_dense = layer
    if src_conv is None or src_dense is None:
        raise RuntimeError("Could not locate Conv2D / Dense in source model")

    conv_kernel, conv_bias = src_conv.get_weights()
    dense_kernel, dense_bias = src_dense.get_weights()

    dst_grid_model.get_layer("perceive").set_weights([conv_kernel, conv_bias])
    dst_grid_model.get_layer("update").set_weights(
        [dense_kernel.reshape(1, 1, *dense_kernel.shape), dense_bias]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/phase4_run4",
                        help="Path to the SavedModel directory.")
    parser.add_argument("--out", default="web/nca.onnx",
                        help="Output .onnx file path.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic", action="store_true",
                        help="Export with dynamic H/W (None) instead of fixed.")
    args = parser.parse_args()

    print(f"Loading source model from {args.weights}")
    src = tf.keras.models.load_model(args.weights)
    src.summary()

    in_shape = src.inputs[0].shape  # (None, 3, 3, C)
    channels = int(in_shape[-1])
    src_conv = next(l for l in src.layers if isinstance(l, tf.keras.layers.Conv2D))
    hidden = src_conv.get_weights()[0].shape[-1]
    print(f"Detected channels={channels}, hidden={hidden}")

    h = None if args.dynamic else args.height
    w = None if args.dynamic else args.width
    grid_model = build_grid_model(channels, hidden, h, w)
    grid_model.summary()
    copy_weights(src, grid_model)

    # Sanity check: per-pixel output of the grid model on a random grid must
    # match the patch model run on the same 3x3 neighbourhood.
    rng = np.random.default_rng(0)
    test_h, test_w = args.height, args.width
    test_grid = rng.standard_normal((1, test_h, test_w, channels)).astype("float32")
    grid_out = grid_model(test_grid).numpy()
    cy, cx = test_h // 2, test_w // 2
    patch = test_grid[:, cy - 1:cy + 2, cx - 1:cx + 2, :]
    patch_out = src(patch).numpy()
    err = float(np.max(np.abs(grid_out[0, cy, cx] - patch_out[0])))
    print(f"Equivalence check max abs error at center pixel: {err:.2e}")
    assert err < 1e-4, "Weight copy did not preserve outputs"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    spec = (tf.TensorSpec(
        (1, h, w, channels) if not args.dynamic else (1, None, None, channels),
        tf.float32, name="grid"),)
    print(f"Converting to ONNX (opset={args.opset}) -> {args.out}")
    model_proto, _ = tf2onnx.convert.from_keras(
        grid_model, input_signature=spec, opset=args.opset, output_path=args.out
    )

    # Write a small metadata json next to the model so the JS side knows the
    # tensor shape and channel layout without hard-coding it.
    import json
    meta = {
        "channels": channels,
        "hidden": hidden,
        "height": args.height,
        "width": args.width,
        "dynamic": args.dynamic,
        "input_name": grid_model.inputs[0].name.split(":")[0],
        "output_name": grid_model.outputs[0].name.split(":")[0],
        "alive_threshold": 0.1,
        "fire_rate": 0.5,
        "rgb_channels": [0, 1, 2],
        "alpha_channel": 3,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
