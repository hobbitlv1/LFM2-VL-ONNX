import argparse
import json
from pathlib import Path
import random

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

ort.preload_dlls(cuda=False, cudnn=False, msvc=True, directory=None)

DEFAULT_ORIGINAL_MODEL = Path("onnx_optimum/model.onnx")
DEFAULT_QUANTIZED_MODEL = Path("quant/model_dynamic.quant.onnx")
DEFAULT_IMAGES_DIR = Path("val2017")
DEFAULT_ANNOTATIONS = Path("annotations/captions_val2017.json")
MODEL_ID = "LiquidAI/LFM2-VL-450M"


class Evaluator:
    def __init__(self, path, proc, ext_prov=False):
        self.path = path
        self.proc = proc

        if ext_prov:
            provs = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            try:
                self.sess = ort.InferenceSession(str(path), providers=provs, sess_options=opts)
            except Exception as e:
                print(f"  Warning: Could not load with extended providers: {e}")
                print(f"  Falling back to CPU only...")
                self.sess = ort.InferenceSession(str(path), providers=['CUDAExecutionProvider'])
        else:
            provs = ['CUDAExecutionProvider']
            self.sess = ort.InferenceSession(str(path), providers=provs)

        self.inp_names = [inp.name for inp in self.sess.get_inputs()]
        print(f"  Model inputs: {self.inp_names}")
        print(f"  Using providers: {self.sess.get_providers()}")

    def prep_inputs(self, img, cap):
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Describe this image: {cap[:50]}..."},
                ],
            },
        ]

        feats = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")

        inputs = {}
        for name in self.inp_names:
            if name in feats:
                tensor = feats[name]
                if name in ['input_ids', 'attention_mask', 'pixel_attention_mask', 'spatial_shapes']:
                    inputs[name] = tensor.numpy().astype(np.int64)
                else:
                    inputs[name] = tensor.numpy().astype(np.float32)

        return inputs

    def run(self, inputs):
        outs = self.sess.run(None, inputs)
        return outs[0]

    def top_preds(self, logits, k=5):
        last = logits[0, -1, :]
        top_idx = np.argsort(last)[-k:][::-1]
        return top_idx


def load_coco(ann_path, img_dir, n_samp=10):
    print(f"Loading annotations from {ann_path}...")
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    imgs = {img['id']: img for img in coco['images']}

    caps_by_img = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in caps_by_img:
            caps_by_img[img_id] = []
        caps_by_img[img_id].append(ann['caption'])

    avail = []
    for img_id, caps in list(caps_by_img.items())[:500]:
        if img_id in imgs:
            info = imgs[img_id]
            pth = img_dir / info['file_name']
            if pth.exists():
                avail.append({
                    'image_id': img_id,
                    'image_path': pth,
                    'file_name': info['file_name'],
                    'captions': caps,
                })

        if len(avail) >= n_samp * 3:
            break

    if len(avail) > n_samp:
        random.seed(420)
        avail = random.sample(avail, n_samp)

    print(f"Selected {len(avail)} images for evaluation")
    return avail


def cmp_preds(orig, quant):
    argmax_match = orig[0] == quant[0]
    top5_match = len(set(orig) & set(quant))
    return argmax_match, top5_match


def softmax(logits):
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.clip(exp_vals.sum(), a_min=1e-12, a_max=None)
    return exp_vals / denom


def dist_metrics(orig_logits, quant_logits):
    orig = orig_logits[0, -1, :]
    quant = quant_logits[0, -1, :]

    orig_p = softmax(orig)
    quant_p = softmax(quant)

    eps = 1e-12
    kl = float(np.sum(orig_p * (np.log(orig_p + eps) - np.log(quant_p + eps))))
    rev_kl = float(np.sum(quant_p * (np.log(quant_p + eps) - np.log(orig_p + eps))))

    denom = (np.linalg.norm(orig) * np.linalg.norm(quant)) + eps
    cos = float(np.dot(orig, quant) / denom)
    mad = float(np.mean(np.abs(orig - quant)))

    return {
        "kl_divergence": kl,
        "reverse_kl_divergence": rev_kl,
        "cosine_similarity": cos,
        "mean_abs_diff": mad,
    }


def eval_models(orig_path, quant_path, samples, proc, verbose=False):
    print("LOADING MODELS")

    print(f"\nLoading original model from {orig_path}...")
    try:
        orig_eval = Evaluator(orig_path, proc, ext_prov=False)
    except Exception as e:
        print(f"Failed to load original model: {e}")
        return {'error': str(e)}

    print(f"\nLoading quantized model from {quant_path}...")
    try:
        quant_eval = Evaluator(quant_path, proc, ext_prov=True)
    except Exception as e:
        print(f"Failed to load quantized model: {e}")
        print(f"\nThe quantized model uses operators (like ConvInteger) that require")
        print(f"    specific ONNX Runtime execution providers or a newer version.")
        print(f"    Consider re-quantizing with different settings or using dynamic quantization.")
        return {'error': str(e)}


    print("RUNNING EVALUATION")

    res = {
        'total_samples': len(samples),
        'argmax_matches': 0,
        'top5_overlaps': [],
        'kl_divergences': [],
        'reverse_kl_divergences': [],
        'cosine_similarities': [],
        'mean_abs_diffs': [],
        'failed_samples': [],
        'sample_details': [],
    }

    for idx, samp in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            img = Image.open(samp['image_path']).convert('RGB')
            cap = samp['captions'][0]

            if verbose:
                print(f"\n[{idx+1}/{len(samples)}] {samp['file_name']}")
                print(f"  Caption: {cap}")

            inputs = orig_eval.prep_inputs(img, cap)

            orig_logits = orig_eval.run(inputs)
            quant_logits = quant_eval.run(inputs)

            orig_top5 = orig_eval.top_preds(orig_logits, k=5)
            quant_top5 = quant_eval.top_preds(quant_logits, k=5)

            metrics = dist_metrics(orig_logits, quant_logits)
            res['kl_divergences'].append(metrics['kl_divergence'])
            res['reverse_kl_divergences'].append(metrics['reverse_kl_divergence'])
            res['cosine_similarities'].append(metrics['cosine_similarity'])
            res['mean_abs_diffs'].append(metrics['mean_abs_diff'])

            match, overlap = cmp_preds(orig_top5, quant_top5)

            if match:
                res['argmax_matches'] += 1

            res['top5_overlaps'].append(overlap)

            orig_tok = proc.tokenizer.decode([orig_top5[0]])
            quant_tok = proc.tokenizer.decode([quant_top5[0]])

            samp_res = {
                'file_name': samp['file_name'],
                'argmax_match': match,
                'top5_overlap': overlap,
                'original_top_token': orig_tok,
                'quantized_top_token': quant_tok,
                'original_top5_ids': orig_top5.tolist(),
                'quantized_top5_ids': quant_top5.tolist(),
                'kl_divergence': metrics['kl_divergence'],
                'reverse_kl_divergence': metrics['reverse_kl_divergence'],
                'cosine_similarity': metrics['cosine_similarity'],
                'mean_abs_diff': metrics['mean_abs_diff'],
            }
            res['sample_details'].append(samp_res)

            if verbose:
                print(f"  Original top token: {orig_tok} (ID: {orig_top5[0]})")
                print(f"  Quantized top token: {quant_tok} (ID: {quant_top5[0]})")
                print(f"  Argmax match: {'Yes' if match else 'No'}")
                print(f"  Top-5 overlap: {overlap}/5")

        except Exception as e:
            print(f"\nError processing {samp['file_name']}: {e}")
            res['failed_samples'].append({'file_name': samp['file_name'], 'error': str(e)})

    return res


def show_results(res):
    print("EVALUATION RESULTS")

    total = res['total_samples']
    matches = res['argmax_matches']
    acc = (matches / total * 100) if total > 0 else 0

    print(f"\nARGMAX ACCURACY:")
    print(f"  Matches: {matches}/{total}")
    print(f"  Accuracy: {acc:.2f}%")

    avg_t5 = np.mean(res['top5_overlaps']) if res['top5_overlaps'] else 0
    print(f"\nTOP-5 OVERLAP:")
    print(f"  Average: {avg_t5:.2f}/5")
    print(f"  This means {avg_t5/5*100:.1f}% of top-5 predictions overlap on average")

    if res['kl_divergences']:
        avg_kl = float(np.mean(res['kl_divergences']))
        avg_rkl = float(np.mean(res['reverse_kl_divergences']))
        avg_cos = float(np.mean(res['cosine_similarities']))
        avg_mad = float(np.mean(res['mean_abs_diffs']))

        print(f"\nDISTRIBUTION SIMILARITY:")
        print(f"  KL(original || quant): {avg_kl:.6f}")
        print(f"  KL(quant || original): {avg_rkl:.6f}")
        print(f"  Cosine similarity: {avg_cos:.6f}")
        print(f"  Mean abs logit diff: {avg_mad:.6f}")

    if res['failed_samples']:
        print(f"\nFAILED SAMPLES: {len(res['failed_samples'])}")
        for fail in res['failed_samples']:
            print(f"  - {fail['file_name']}: {fail['error']}")

    misses = [s for s in res['sample_details'] if not s['argmax_match']]
    if misses:
        print(f"\nMISMATCHED PREDICTIONS (showing first 5):")
        for samp in misses[:5]:
            print(f"  {samp['file_name']}:")
            print(f"    Original:  {samp['original_top_token']} (ID: {samp['original_top5_ids'][0]})")
            print(f"    Quantized: {samp['quantized_top_token']} (ID: {samp['quantized_top5_ids'][0]})")
            print(f"    Top-5 overlap: {samp['top5_overlap']}/5")
            print(
                "    Distribution: KL={:.6f}, Reverse KL={:.6f}, Cosine={:.6f}, Mean|D|={:.6f}".format(
                    samp['kl_divergence'], samp['reverse_kl_divergence'],
                    samp['cosine_similarity'], samp['mean_abs_diff'],
                )
            )



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original vs quantized ONNX models on COCO validation set")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL_MODEL, help="Path to original ONNX model")
    parser.add_argument("--quantized", type=Path, default=DEFAULT_QUANTIZED_MODEL, help="Path to quantized ONNX model")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Path to images directory")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="Path to captions_val2017.json")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of images to evaluate (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress for each sample")
    parser.add_argument("--save-results", type=Path, help="Save detailed results to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.original.exists():
        print(f"Error: Original model not found at {args.original}")
        return

    if not args.quantized.exists():
        print(f"Error: Quantized model not found at {args.quantized}")
        return

    if not args.images_dir.exists():
        print(f"Error: Images directory not found at {args.images_dir}")
        return

    if not args.annotations.exists():
        print(f"Error: Annotations file not found at {args.annotations}")
        return

    print("Loading processor...")
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    samples = load_coco(args.annotations, args.images_dir, args.num_samples)

    if not samples:
        print(" Error: No valid samples found")
        return

    res = eval_models(args.original, args.quantized, samples, proc, verbose=args.verbose)

    show_results(res)

    if args.save_results:
        import json
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nDetailed results saved to {args.save_results}")


if __name__ == "__main__":
    main()
