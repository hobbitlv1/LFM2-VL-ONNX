import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor
from pathlib import Path

MODEL_ID = "LiquidAI/LFM2-VL-450M"

print("Loading processor")
proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading test image")
img_path = Path("val2017/000000000139.jpg")
img = Image.open(img_path).convert('RGB')

print("Preparing inputs...")
conv = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image"},
        ],
    },
]

feats = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")

inputs = {}
for name, tensor in feats.items():
    if name in ['input_ids', 'attention_mask', 'pixel_attention_mask', 'spatial_shapes']:
        inputs[name] = tensor.numpy().astype(np.int64)
    else:
        inputs[name] = tensor.numpy().astype(np.float32)

models = [
    ("Original", "onnx_optimum/model.onnx"),
    ("Dynamic Quant", "quant/model_dynamic.quant.onnx"),
    ("Static Quant", "quant/model_static.quant.onnx"),
]

print("COMPARING LOGITS")


orig_logits = None
res = {}

for name, path in models:
    try:
        print(f"\n{name}: {path}")
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        outs = sess.run(None, inputs)
        logits = outs[0]

        last = logits[0, -1, :]

        print(f"Logits shape: {logits.shape}")
        print(f"Last token logits stats:")
        print(f"Min: {last.min():.6f}")
        print(f"Max: {last.max():.6f}")
        print(f"Mean: {last.mean():.6f}")
        print(f"Std: {last.std():.6f}")

        top5_idx = np.argsort(last)[-5:][::-1]
        top5_scores = last[top5_idx]

        print(f"Top 5 predictions:")
        for i, (idx, score) in enumerate(zip(top5_idx, top5_scores)):
            tok = proc.tokenizer.decode([idx])
            print(f"{i+1}. Token: '{tok}' (ID: {idx}, Score: {score:.6f})")

        res[name] = {
            'logits': last,
            'top5_indices': top5_idx,
            'top5_scores': top5_scores,
        }

        if name == "Original":
            orig_logits = last

    except Exception as e:
        print(f"Error: {e}")

if orig_logits is not None:

    print("COMPARISON TO ORIGINAL")


    for name in ["Dynamic Quant", "Static Quant"]:
        if name in res:
            q_logits = res[name]['logits']

            print(f"\n{name}:")

            abs_diff = np.abs(orig_logits - q_logits)
            rel_diff = abs_diff / (np.abs(orig_logits) + 1e-8)

            print(f"Absolute difference:")
            print(f"Mean: {abs_diff.mean():.6f}")
            print(f"Max: {abs_diff.max():.6f}")
            print(f"Std: {abs_diff.std():.6f}")

            print(f"Relative difference:")
            print(f"Mean: {rel_diff.mean():.6f}")
            print(f"Max: {rel_diff.max():.6f}")

            o_top5 = res["Original"]['top5_indices']
            q_top5 = res[name]['top5_indices']

            print(f"Top-5 overlap: {len(set(o_top5) & set(q_top5))}/5")
            print(f"Argmax match: {o_top5[0] == q_top5[0]}")

            if np.any(np.isnan(q_logits)):
                print(f"WARNING: NaN values detected!")
            if np.any(np.isinf(q_logits)):
                print(f"WARNING: Inf values detected!")

            if q_logits.max() > 100:
                print(f"WARNING: Very large logits detected (max: {q_logits.max():.2f})")
            if abs_diff.mean() > 1.0:
                print(f"WARNING: Large mean difference ({abs_diff.mean():.2f})")
