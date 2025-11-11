import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor

MODEL_ID = "LiquidAI/LFM2-VL-450M"
DEFAULT_LIMIT = 10
ANNOTATIONS_PATH = Path("annotations/captions_val2017.json")
IMAGES_DIR = Path("val2017")
OUTPUT_PATH = Path("quant/validation_dataset.pt")


class SampleMeta:
    def __init__(self, image_id, file_name, caption):
        self.image_id = image_id
        self.file_name = file_name
        self.caption = caption


def load_ds(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_lookup(images):
    return {int(img["id"]): img["file_name"] for img in images}


def mk_conv(image, caption):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": caption},
            ],
        }
    ]


def load_img(path):
    with Image.open(path) as f:
        return f.convert("RGB")


def tok_key(features):
    input_ids = features["input_ids"]
    att_mask = features["attention_mask"]
    pix_mask = features["pixel_attention_mask"]

    txt_len = int(input_ids.shape[-1])
    txt_toks = int(att_mask.sum().item())
    pix_toks = int(pix_mask.sum().item())

    return (txt_len, txt_toks, pix_toks)


def proc_samples(proc, annotations, lookup, img_dir, limit):
    if limit <= 0:
        return [], None, Counter(), 0

    candidates = defaultdict(list)
    freqs = Counter()
    target = None

    for entry in annotations:
        img_id = int(entry["image_id"])
        cap = entry["caption"].strip()

        fname = lookup.get(img_id)
        if not fname:
            continue

        img_path = img_dir / fname
        if not img_path.exists():
            continue

        img = load_img(img_path)
        conv = mk_conv(img, cap)
        feats = proc.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        key = tok_key(feats)
        bucket = candidates[key]
        if len(bucket) < limit:
            bucket.append(SampleMeta(image_id=img_id, file_name=fname, caption=cap))

        freqs[key] += 1
        if target is None and freqs[key] >= limit and len(bucket) >= limit:
            target = key
            break

    if target is None:
        if not freqs:
            return [], None, freqs, 0
        target = freqs.most_common(1)[0][0]

    sel_meta = candidates.get(target, [])
    samples = []

    for meta in sel_meta:
        img_path = img_dir / meta.file_name
        if not img_path.exists():
            continue

        img = load_img(img_path)
        conv = mk_conv(img, meta.caption)
        feats = proc.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        tensors = OrderedDict()
        for name, tensor in feats.items():
            tensors[name] = tensor.detach().cpu()

        samples.append(
            {
                "image_id": meta.image_id,
                "file_name": meta.file_name,
                "caption": meta.caption,
                "inputs": tensors,
            }
        )
        if len(samples) >= limit:
            break

    dups = 0
    if samples and len(samples) < limit:
        base = list(samples)
        while len(samples) < limit:
            src = base[len(samples) % len(base)]
            cloned = OrderedDict((n, t.clone()) for n, t in src["inputs"].items())
            samples.append(
                {
                    "image_id": src["image_id"],
                    "file_name": src["file_name"],
                    "caption": src["caption"],
                    "inputs": cloned,
                    "duplicate": True,
                }
            )
            dups += 1

    if not samples:
        return [], target, freqs, dups

    if len(samples) > limit:
        samples = samples[:limit]

    if dups:
        print(f"Warning: Added {dups} duplicated samples to reach {limit} items.")

    return samples, target, freqs, dups


def save_val(samples, out_path, model_id, key=None, freqs=None, dups=0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_id": model_id,
        "num_samples": len(samples),
        "samples": samples,
    }

    if key is not None:
        data["token_key"] = {
            "text_sequence_length": key[0],
            "text_token_count": key[1],
            "pixel_token_count": key[2],
        }
    if freqs:
        freq_dict = {
            f"text_len={k[0]}|text_tokens={k[1]}|pixel_tokens={k[2]}": cnt
            for k, cnt in freqs.items()
        }
        data["token_frequencies"] = freq_dict
    if dups:
        data["duplicates_added"] = dups

    torch.save(data, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare validation tensors for ONNX quantization.")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_ds(args.annotations)
    annotations = ds.get("annotations", [])
    lookup = build_lookup(ds.get("images", []))

    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    samples, key, freqs, dups = proc_samples(proc, annotations, lookup, args.images_dir, args.limit)
    if not samples:
        raise RuntimeError("Unable to assemble validation samples with consistent token shapes.")

    save_val(samples, args.output, args.model_id, key, freqs, dups)

    print(f"Saved {len(samples)} samples to {args.output}")
    if key:
        print(f"Token profile: text sequence length={key[0]}, text tokens={key[1]}, pixel tokens={key[2]}")
    if dups:
        print(f"Included {dups} duplicated samples to fulfill the requested limit of {args.limit}.")


if __name__ == "__main__":
    main()
