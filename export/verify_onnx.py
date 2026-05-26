"""Run the exported ONNX model in-Python to verify a full NCA roll-out.

This is the same loop the browser will run, but on CPU with `onnxruntime`,
so we can save the final frame to disk and visually confirm the export.
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image


def step(session, state, alive_threshold, fire_rate, rng, input_name):
    update = session.run(None, {input_name: state})[0]

    # alive mask: a 3x3 max pool over the alpha channel must exceed threshold
    alpha = state[..., 3:4]
    h, w = alpha.shape[1], alpha.shape[2]
    padded = np.pad(alpha, ((0, 0), (1, 1), (1, 1), (0, 0)))
    pooled = np.zeros_like(alpha)
    for dy in range(3):
        for dx in range(3):
            pooled = np.maximum(pooled, padded[:, dy:dy + h, dx:dx + w, :])
    alive = (pooled > alive_threshold).astype(np.float32)

    fire = (rng.random(alpha.shape) < fire_rate).astype(np.float32)
    return state + update * alive * fire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="web/nca.onnx")
    parser.add_argument("--meta", default="web/nca.json")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default="export/onnx_final_frame.png")
    args = parser.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    h, w, c = meta["height"], meta["width"], meta["channels"]
    state = np.zeros((1, h, w, c), dtype=np.float32)
    state[0, h // 2, w // 2, 3:] = 1.0  # seed center cell alive across all hidden channels

    rng = np.random.default_rng(0)
    for i in range(args.steps):
        state = step(session, state, meta["alive_threshold"], meta["fire_rate"], rng, input_name)
        alive_pixels = int(((state[..., 3] > 0.1).sum()))
        if i % 20 == 0:
            print(f"step {i:3d} alive={alive_pixels}")

    rgb = np.clip(state[0, :, :, :3], 0, 1)
    alpha = np.clip(state[0, :, :, 3:4], 0, 1)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    img = (rgba * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(args.out)
    print(f"Saved final frame -> {args.out}")


if __name__ == "__main__":
    main()
