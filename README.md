# cellularimages

A Growing Neural Cellular Automata model — trained in TensorFlow, deployed to the
browser via ONNX + WebGL.

The training code lives in `main.ipynb` (see the original
[Mordvintsev et al. 2020](https://distill.pub/2020/growing-ca/) paper).
This repo adds a deployment pipeline so the trained model can run entirely
client-side in any modern browser, with GPU acceleration via WebGL.

## Browser demo

The `web/` folder is a static site:

```
web/
  index.html   # canvas + controls
  nca.js       # inference loop (ort.InferenceSession)
  nca.onnx     # exported model
  nca.json     # shape / channel metadata
```

Serve it from anywhere:

```bash
cd web && python3 -m http.server 8000
# open http://localhost:8000
```

Click the canvas to damage cells and watch the model regrow the target.
The Backend dropdown switches between WebGL (GPU) and WASM (CPU) execution
providers of `onnxruntime-web`; WebGL is the default.

## Re-exporting the model

If you retrain in `main.ipynb` and want to publish a new `nca.onnx`:

```bash
pip install -r export/requirements.txt
python export/export_to_onnx.py \
    --weights weights/phase4_run4 \
    --out web/nca.onnx \
    --height 64 --width 96
```

The script does two things:

1. **Rewraps the patch-based update model as a fully-convolutional grid model.**
   The training network takes a single (3, 3, C) patch and emits a (C,) update;
   it is mathematically identical to a `Conv2D(hidden, 3×3, same) → ReLU →
   Conv2D(C, 1×1)` stack applied to the whole grid. The script copies the
   trained weights into that form and asserts bit-exact equivalence at a center
   pixel.
2. **Exports to ONNX via `tf2onnx`** at opset 13. The resulting graph only
   contains `Conv` and `Relu` — both fully supported by `onnxruntime-web`'s
   WebGL execution provider.

`export/verify_onnx.py` runs the same per-step NCA loop the browser runs, but
on CPU via `onnxruntime`, and saves the final frame to
`export/onnx_final_frame.png`. Useful as a smoke test before publishing.

## How inference is structured

The ONNX graph is intentionally minimal — just the two conv layers. The
stochastic fire mask and the 3×3 max-pool alive mask are computed in JS each
step (no random ops in the graph, so the WebGL backend stays happy):

```
for each frame:
    delta  = model(state)              # ONNX / WebGL
    alive  = maxPool3x3(state.alpha) > 0.1
    fire   = random() < fireRate
    state += delta * alive * fire      # in-place on a Float32Array
    draw(state[:, :, 0:4]) to <canvas>
```
