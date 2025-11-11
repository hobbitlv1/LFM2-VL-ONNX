from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from contextlib import contextmanager
import functools
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
import onnx

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from transformers.models.siglip2 import modeling_siglip2

from optimum.exporters.onnx import OnnxConfigWithPast
from optimum.exporters.onnx.convert import onnx_export_from_model
from optimum.exporters.tasks import TasksManager
from optimum.utils import (
    NormalizedTextAndVisionConfig,
    DummyDecoderTextInputGenerator,
    DummyPastKeyValuesGenerator,
    DummyVisionInputGenerator,
)
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


MODEL_ID = "LiquidAI/LFM2-VL-450M"
IMAGE_URL = "https://www.ilankelman.org/stopsigns/australia.jpg"
PROMPT = "What is in this image?"
EXPORT_DIR = Path("onnx_optimum")
DEFAULT_OPSET = 20  # conservative default
MAX_VISION_TOKENS_OVERRIDE: int | None = None
_MAX_VISION_OVERRIDE_LOGGED = False

log = logging.getLogger("lfm2vl_export")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def convert_dtypes(tensors, dtypes):
    for k, dtype in dtypes.items():
        if k in tensors:
            tensors[k] = tensors[k].to(dtype=dtype)


def load_model(cls, model_id, rev=None, **kw):
    return cls.from_pretrained(model_id, trust_remote_code=True, revision=rev, **kw)


def build_opts(imax, merge, rank, v):
    return {"int_max": imax, "auto_merge": merge, "guess_output_rank": rank, "verbose": v}


def add_bool_arg(parser, name, default, help_txt):
    parser.add_argument(name, type=parse_bool, default=default, metavar="{true,false}", help=help_txt)


def resize_pos_emb(pos_emb, spatial_shapes, max_length):
    global _MAX_VISION_OVERRIDE_LOGGED
    if MAX_VISION_TOKENS_OVERRIDE and MAX_VISION_TOKENS_OVERRIDE > 0:
        max_length = MAX_VISION_TOKENS_OVERRIDE
        if not _MAX_VISION_OVERRIDE_LOGGED:
            log.info("Overriding max vision tokens cap -> %s", max_length)
            _MAX_VISION_OVERRIDE_LOGGED = True

    B = spatial_shapes.shape[0]
    C = pos_emb.shape[-1]
    src_dtype = pos_emb.dtype

    pos = pos_emb.permute(2, 0, 1).unsqueeze(0)
    if pos.dtype not in (torch.float32, torch.float64):
        pos = pos.to(torch.float32)

    out = pos_emb.new_empty((int(B), int(max_length), C))

    for i in range(int(B)):
        hi = int(spatial_shapes[i, 0].item())
        wi = int(spatial_shapes[i, 1].item())
        resized = torch.nn.functional.interpolate(
            pos, size=(hi, wi), mode="bilinear", align_corners=True, antialias=False
        )
        flat = resized.reshape(C, hi * wi).transpose(0, 1).to(src_dtype)
        take = min(int(max_length), flat.shape[0])
        out[i, :take] = flat[:take]
        fill = int(max_length) - take
        if fill > 0:
            out[i, take:take + fill] = flat[take - 1].unsqueeze(0).expand(fill, -1)
    return out


@contextmanager
def patched_siglip2_resize(enable=True):
    if not enable:
        yield
        return
    cls = modeling_siglip2.Siglip2VisionEmbeddings
    orig = getattr(cls, "resize_positional_embeddings", None)
    cls.resize_positional_embeddings = staticmethod(resize_pos_emb)
    try:
        log.info("Applied SigLIP2 resize positional embeddings export patch")
        yield
    finally:
        if orig is not None:
            cls.resize_positional_embeddings = orig
        log.info("Restored SigLIP2 resize")


class Lfm2VlNormalizedConfig(NormalizedTextAndVisionConfig):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, allow_new=True, **kwargs)

    @property
    def _tc(self):
        return getattr(self.config, "text_config", None)

    def _tc_get(self, name, default=None):
        tc = self._tc
        return getattr(tc, name, default) if tc is not None else default

    def get_req(self, names, desc):
        for name in names:
            val = self._tc_get(name, None)
            if val is not None:
                return int(val)
        raise AttributeError(f"{desc} not found in text_config")

    @property
    def num_layers(self):
        layer_types = self._tc_get("layer_types", None)
        if isinstance(layer_types, (list, tuple)):
            count = sum(1 for t in layer_types if "attention" in str(t))
            if count > 0:
                return int(count)
        nl = self._tc_get("num_hidden_layers", None)
        if nl is None:
            raise AttributeError("num_layers could not be derived from text_config")
        return int(nl)

    @property
    def hidden_size(self):
        return self.get_req(["hidden_size"], "hidden_size")

    @property
    def num_attention_heads(self):
        return self.get_req(["num_attention_heads", "num_heads"], "num_attention_heads")

    @property
    def num_key_value_heads(self):
        nk = self._tc_get("num_key_value_heads", None)
        if nk is not None:
            return int(nk)
        return self.num_attention_heads


class Lfm2VlOnnxConfig(OnnxConfigWithPast):
    NORMALIZED_CONFIG_CLASS = Lfm2VlNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyDecoderTextInputGenerator,
        DummyVisionInputGenerator,
        DummyPastKeyValuesGenerator,
    )
    DEFAULT_ONNX_OPSET = DEFAULT_OPSET

    @property
    def inputs(self):
        inputs = OrderedDict()
        inputs["input_ids"] = {0: "batch_size", 1: "sequence_length"}
        inputs["attention_mask"] = {0: "batch_size", 1: "sequence_length"}
        inputs["pixel_values"] = {0: "batch_size", 1: "max_vision_tokens", 2: "vision_hidden"}
        inputs["spatial_shapes"] = {0: "num_images", 1: "two"}
        inputs["pixel_attention_mask"] = {0: "batch_size", 1: "max_vision_tokens"}
        if self.use_past_in_inputs:
            self.add_past_key_values(inputs, direction="inputs")
        return inputs

    @property
    def outputs(self):
        outputs = OrderedDict()
        outputs["logits"] = {0: "batch_size", 1: "sequence_length", 2: "vocab_size"}
        if self.use_past:
            self.add_past_key_values(outputs, direction="outputs")
        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        if not self._preprocessors:
            return super().generate_dummy_inputs(framework=framework, **kwargs)

        if framework != "pt":
            raise ValueError("Only PyTorch dummy inputs are supported for this custom config.")

        processor = self._preprocessors[0]
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        # Synthetic image so export does not depend on external assets.
        dummy_image = Image.new("RGB", (width, height), color=(128, 128, 128))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ]
        features = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        tensor_inputs = {k: v for k, v in features.items()}
        convert_dtypes(tensor_inputs, {
            "input_ids": torch.int64,
            "attention_mask": torch.int64,
            "pixel_values": torch.float32,
            "pixel_attention_mask": torch.int64,
            "spatial_shapes": torch.int64,
        })

        # If a cap override is set, pad/truncate to that length so ONNX graph capacity matches.
        if (
            MAX_VISION_TOKENS_OVERRIDE
            and "pixel_values" in tensor_inputs
            and "pixel_attention_mask" in tensor_inputs
        ):
            pv = tensor_inputs["pixel_values"]          # [B, T, H]
            pam = tensor_inputs["pixel_attention_mask"] # [B, T]
            if pv.dim() == 3 and pam.dim() == 2:
                B, T, H = pv.shape
                target = MAX_VISION_TOKENS_OVERRIDE
                if T < target:
                    pad_len = target - T
                    pad_vals = torch.zeros((B, pad_len, H), dtype=pv.dtype, device=pv.device)
                    pad_mask = torch.zeros((B, pad_len), dtype=pam.dtype, device=pam.device)
                    tensor_inputs["pixel_values"] = torch.cat([pv, pad_vals], dim=1)
                    tensor_inputs["pixel_attention_mask"] = torch.cat([pam, pad_mask], dim=1)
                    log.info("Padded dummy pixel tensors to %d tokens for export", target)
                elif T > target:
                    tensor_inputs["pixel_values"] = pv[:, :target, :]
                    tensor_inputs["pixel_attention_mask"] = pam[:, :target]
                    log.info("Truncated dummy pixel tensors to %d tokens for export", target)

        ordered_inputs = OrderedDict((name, tensor_inputs[name]) for name in self.inputs)
        return ordered_inputs

    def patch_model_for_export(self, model, model_kwargs: dict | None = None):
        # Align use_cache with use_past to avoid mismatched graph signatures.
        model.config.use_cache = bool(self.use_past)
        # Ensure eager attention impl for export stability
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        base_cm = super().patch_model_for_export(model, model_kwargs=model_kwargs)

        # Wrap the already patched forward to convert HybridConvCache to a flat tuple of tensors
        @contextmanager
        def _wrap_forward():
            with base_cm:
                orig_forward = getattr(model, "forward")
                output_names = list(self.outputs.keys())

                @functools.wraps(orig_forward)
                def _forward(*args, **kwargs):
                    out = orig_forward(*args, **kwargs)

                    # Expect a dict from Optimum's ModelPatcher; fallback to passthrough otherwise.
                    if not isinstance(out, dict):
                        return out

                    pkv = out.get("past_key_values", None)
                    present_pairs = None

                    if pkv is not None:
                        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                            try:
                                layer_types = getattr(pkv, "layer_types", None)
                                present_pairs = []
                                if isinstance(layer_types, (list, tuple)) and layer_types:
                                    for idx, typ in enumerate(layer_types):
                                        if "attention" in str(typ):
                                            k = pkv.key_cache[idx] if idx < len(pkv.key_cache) else None
                                            v = pkv.value_cache[idx] if idx < len(pkv.value_cache) else None
                                            if k is not None and v is not None:
                                                present_pairs.append((k, v))
                                else:
                                    # Fallback best-effort: use all non-empty entries
                                    for idx in range(len(getattr(pkv, "key_cache", []))):
                                        k = pkv.key_cache[idx]
                                        v = pkv.value_cache[idx]
                                        if isinstance(k, torch.Tensor) and k.numel() > 0:
                                            present_pairs.append((k, v))
                            except Exception:
                                present_pairs = None
                        elif isinstance(pkv, (list, tuple)):
                            present_pairs = list(pkv)

                    # If our ONNX expects present.* outputs, build an ordered dict following output_names order.
                    if any(name.startswith("present") for name in output_names):
                        ordered = OrderedDict()
                        for name in output_names:
                            if name == "logits":
                                ordered[name] = out.get("logits")
                            elif name.startswith("present") and present_pairs is not None:
                                try:
                                    _, idx_str, kind = name.split(".")
                                    idx = int(idx_str)
                                    kv = present_pairs[idx]
                                    ordered[name] = kv[0] if kind == "key" else kv[1]
                                except Exception:
                                    ordered[name] = out.get(name)
                            else:
                                ordered[name] = out.get(name)
                        # Remove unsupported nested pkv entry if present
                        if "past_key_values" in ordered:
                            ordered.pop("past_key_values", None)
                        return ordered

                    return out

                setattr(model, "forward", _forward)
                try:
                    yield
                finally:
                    setattr(model, "forward", orig_forward)

        return _wrap_forward()

    @staticmethod
    def get_onnx_input_names():
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "spatial_shapes",
            "pixel_attention_mask",
        ]


def register_custom_config(model_type: str = "lfm2_vl") -> None:
    register_for_onnx = TasksManager.create_register("onnx", overwrite_existing=True)

    @register_for_onnx(model_type, "image-text-to-text")  # positional: (model_type, *tasks)
    class _Lfm2VlOnnxConfig(Lfm2VlOnnxConfig):
        pass

    log.info("Registered custom ONNX config via TasksManager for model_type='%s', task='image-text-to-text'",
             model_type)


def run_sym_infer(model, options):
    return SymbolicShapeInference.infer_shapes(
        model,
        int_max=int(options.get("int_max", 2_147_483_647)),
        auto_merge=bool(options.get("auto_merge", True)),
        guess_output_rank=bool(options.get("guess_output_rank", True)),
        verbose=int(options.get("verbose", 0)),
    )


def run_shape_infer(onnx_path, run_shape_inference=False,
                    run_symbolic_shape_inference=False, symbolic_options=None):

    if not (run_shape_inference or run_symbolic_shape_inference):
        log.info("Skipping shape inference (dynamic shapes preserved)")
        return

    options = symbolic_options or {}

    try:
        model = onnx.load(str(onnx_path))
        out_path = onnx_path.with_name(onnx_path.stem + ".inferred.onnx")
        inferred = None

        if run_symbolic_shape_inference:
            log.info("Running ORT symbolic shape inference...")
            try:
                inferred = run_sym_infer(model, options)
            except Exception as sym_err:
                log.warning("Symbolic shape inference failed: %s", sym_err)
                if not run_shape_inference:
                    log.warning("Re-run with --run-shape-inference to fall back to onnx.shape_inference.")
                    raise
                log.info("Falling back to onnx.shape_inference...")

        if inferred is None:
            log.info("Running ONNX shape inference...")
            inferred = onnx.shape_inference.infer_shapes(model)

        onnx.save(inferred, str(out_path))
        log.info("Wrote inferred model: %s", out_path)
    except Exception as e:
        log.error("Shape inference failed: %s", e)


def ort_preprocess(inp, out, skip_symbolic_shape=True, skip_onnx_shape=False,
                   skip_optimization=False, auto_merge=True, guess_output_rank=True,
                   verbose=1, int_max=2_147_483_647):
    import importlib
    import inspect

    fn = None
    try:
        mod = importlib.import_module("onnxruntime.quantization.preprocess")
        for name in ("preprocess", "preprocess_model", "quant_pre_process"):
            cand = getattr(mod, name, None)
            if callable(cand):
                fn = cand
                log.info("Using onnxruntime.quantization.preprocess.%s", name)
                break
    except Exception as e:
        log.warning("preprocess module not available: %s", e)
    if fn is None:
        mod2 = importlib.import_module("onnxruntime.quantization.shape_inference")
        fn = getattr(mod2, "quant_pre_process")
        log.info("Using onnxruntime.quantization.shape_inference.quant_pre_process")

    sig = inspect.signature(fn)
    kwargs = {}
    param_map = {
        "skip_symbolic_shape": skip_symbolic_shape,
        "skip_onnx_shape": skip_onnx_shape,
        "skip_optimization": skip_optimization,
        "auto_merge": auto_merge,
        "guess_output_rank": guess_output_rank,
        "verbose": verbose,
        "int_max": int_max,
    }
    for key, value in param_map.items():
        if key in sig.parameters:
            kwargs[key] = value

    fn(str(inp), str(out), **kwargs)
    log.info("Wrote: %s", out)
    return out


def run_post_export_preprocessing(
    onnx_path: Path,
    export_dir: Path,
    method: str = "ort",
    preprocess_kwargs: dict | None = None,
) -> None:
    # Run pre-processing right after export, before quantization.
    method = (method or "none").lower()
    if method not in {"none", "ort"}:
        log.warning("Unknown preprocessing method %r; skipping.", method)
        return

    # Prefer shape-inferred model if present
    inferred = onnx_path.with_name(onnx_path.stem + ".inferred.onnx")
    src = inferred if inferred.exists() else onnx_path

    out_dir = export_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "ort":
        ort_out = out_dir / "model.pre.ort.onnx"
        try:
            ort_preprocess(src, ort_out, **(preprocess_kwargs or {}))
        except Exception as e:
            log.error("Preprocess failed: %s", e)


def check_and_externalize(path, ext=False, threshold=0):
    try:
        onnx.checker.check_model(str(path))
        log.info("onnx.checker.check_model passed for %s", path)
    except Exception as e:
        log.error("onnx.checker.check_model failed: %s", e)
        return

    if ext:
        try:
            model = onnx.load(str(path))
            from onnx.external_data_helper import convert_model_to_external_data
            convert_model_to_external_data(model, all_tensors_to_one_file=True, size_threshold=threshold)
            onnx.save(model, str(path))
            log.info("Converted to external data (.onnx + .data where applicable)")
        except Exception as e:
            log.error("Externalization failed: %s", e)


def parse_args():
    parser = argparse.ArgumentParser(description="Export LFM2-VL models to ONNX.")
    parser.add_argument("--image", type=str, default=IMAGE_URL, help="Image path or URL for the sample prompt.")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt to pair with the sample image.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=EXPORT_DIR,
        help="Output directory for ONNX artifacts (default: onnx_optimum).",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="Target ONNX opset version.")
    parser.add_argument(
        "--max-vision-tokens",
        type=int,
        default=None,
        help="Override the maximum number of vision tokens (applies to config and positional embeddings).",
    )
    parser.add_argument(
        "--preprocess-method",
        choices=["none", "ort"],
        default="ort",
        help="Post-export preprocessing strategy.",
    )
    add_bool_arg(parser, "--preprocess-skip-symbolic-shape", True,
                 "Skip symbolic shape inference during ORT preprocessing (default: true).")
    add_bool_arg(parser, "--preprocess-skip-onnx-shape", False,
                 "Skip ONNX shape inference inside ORT preprocessing (default: false).")
    add_bool_arg(parser, "--preprocess-skip-optimization", False,
                 "Skip ORT graph optimizations during preprocessing (default: false).")
    add_bool_arg(parser, "--preprocess-auto-merge", True,
                 "Allow ORT preprocess auto-merge heuristics when supported (default: true).")
    add_bool_arg(parser, "--preprocess-guess-output-rank", True,
                 "Allow ORT preprocess output-rank guessing when supported (default: true).")
    parser.add_argument(
        "--preprocess-verbose",
        type=int,
        default=1,
        help="Verbosity level for ORT preprocessing (default: 1).",
    )
    parser.add_argument(
        "--preprocess-int-max",
        type=int,
        default=2_147_483_647,
        help="Maximum integer bound for ORT preprocessing symbolic inference (default: 2_147_483_647).",
    )
    parser.add_argument(
        "--disable-resize-patch",
        action="store_true",
        help="Use the stock SigLIP2 resize implementation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=45,
        help="Number of tokens to generate for the sample decode.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Export with cache-enabled graph I/O (KV cache).",
    )
    parser.add_argument(
        "--run-shape-inference",
        action="store_true",
        help="Run standard ONNX shape inference after export.",
    )
    parser.add_argument(
        "--run-symbolic-shape-inference",
        action="store_true",
        help="Run ORT symbolic shape inference after export.",
    )
    parser.add_argument(
        "--symbolic-int-max",
        type=int,
        default=2_147_483_647,
        help="Maximum integer bound for symbolic shape inference (default: 2_147_483_647).",
    )
    add_bool_arg(parser, "--symbolic-auto-merge", True,
                 "Enable auto-merge during symbolic shape inference (default: true).")
    add_bool_arg(parser, "--symbolic-guess-output-rank", True,
                 "Enable output-rank guessing during symbolic shape inference (default: true).")
    parser.add_argument(
        "--symbolic-verbose",
        type=int,
        default=0,
        help="Verbosity level for symbolic shape inference (default: 0).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch, tag, or commit) to pin when using trust_remote_code=True.",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Save weights in external data format after export (creates .data sidecar).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    global MAX_VISION_TOKENS_OVERRIDE
    MAX_VISION_TOKENS_OVERRIDE = args.max_vision_tokens
    Lfm2VlOnnxConfig.DEFAULT_ONNX_OPSET = args.opset

    preprocess_kwargs = {
        "skip_symbolic_shape": args.preprocess_skip_symbolic_shape,
        "skip_onnx_shape": args.preprocess_skip_onnx_shape,
        "skip_optimization": args.preprocess_skip_optimization,
        **build_opts(args.preprocess_int_max, args.preprocess_auto_merge,
                     args.preprocess_guess_output_rank, args.preprocess_verbose),
    }

    register_custom_config(model_type="lfm2_vl")

    model = load_model(AutoModelForImageTextToText, MODEL_ID, args.revision)
    processor = load_model(AutoProcessor, MODEL_ID, args.revision)
    model.eval()

    raise_vision_cap(model, args.max_vision_tokens)

    # Demo generation
    try:
        image = load_image(args.image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=args.use_cache)
        log.info("Sample decode: %s", processor.batch_decode(outputs, skip_special_tokens=True)[0])
    except Exception as e:
        log.warning("Sample generation failed: %s", e)

    export_dir = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    config = load_model(AutoConfig, MODEL_ID, args.revision)
    onnx_config = Lfm2VlOnnxConfig(
        config,
        task="image-text-to-text",
        use_past=args.use_cache, 
        preprocessors=[processor],
    )

    # Re-tie weights to reduce duplicated initializers
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
            log.info("Re-tied model weights before export")
        except Exception as _tw_err:
            log.warning("Skipping tie_weights due to: %s", _tw_err)

    onnx_path = export_dir / "model.onnx"

    # Scoped patch for SigLIP2 only during export
    with patched_siglip2_resize(enable=not args.disable_resize_patch):
        onnx_export_from_model(
            model=model,
            output=str(export_dir),
            opset=args.opset,
            task="image-text-to-text",
            custom_onnx_configs={"model": onnx_config},
            no_post_process=False,
            do_validation=True,
            model_kwargs={"use_cache": args.use_cache},
            preprocessors=[processor],
            atol=1e-2,
        )

    log.info("Model exported to: %s", onnx_path)

    run_shape_infer(onnx_path, args.run_shape_inference,
                    args.run_symbolic_shape_inference,
                    build_opts(args.symbolic_int_max, args.symbolic_auto_merge,
                              args.symbolic_guess_output_rank, args.symbolic_verbose))

    # Pre-processing (before quantization)
    run_post_export_preprocessing(
        onnx_path,
        export_dir,
        method=args.preprocess_method,
        preprocess_kwargs=preprocess_kwargs,
    )

    # Check & optionally externalize
    check_and_externalize(onnx_path, args.external_data, 0)


def update_attrs(cfg, names, val):
    updated = []
    for name in names:
        if hasattr(cfg, name):
            try:
                old = getattr(cfg, name)
                setattr(cfg, name, val)
                updated.append((name, old, val))
            except Exception:
                pass
    return updated


def raise_vision_cap(model, cap=None):
    if not cap or cap <= 0:
        return

    cfg = getattr(model, "config", None)
    if cfg is None:
        return

    candidates = [
        "max_image_tokens", "vision_max_positions", "max_num_visual_features",
        "image_feature_max_len", "image_positions", "image_embedding_length",
        "vision_tokens", "vision_max_tokens",
    ]
    updated = update_attrs(cfg, candidates, cap)

    if updated:
        log.info("Raised config caps: %s", ", ".join(f"{n}:{o}->{c}" for n,o,c in updated))


if __name__ == "__main__":
    main()
