from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image

from transformers import AutoProcessor
from onnxruntime.quantization.calibrate import CalibrationDataReader

# Reuse the ONNX input names defined in the custom export config
try:
    from import_and_export import Lfm2VlOnnxConfig 
    ONNX_INPUT_NAMES = Lfm2VlOnnxConfig.get_onnx_input_names()
except Exception:
    # Fallback to the known interface if import-time deps are heavy/missing
    ONNX_INPUT_NAMES = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "spatial_shapes",
        "pixel_attention_mask",
    ]


class Lfm2VlCalibrationDataReader(CalibrationDataReader):
    #Validation data reader for LiquidAI/LFM2-VL-450M ONNX model.

    #It takes a folder of images and a fixed text prompt and produces multimodal
    #inputs that match the ONNX model's expected input names.

    def __init__(
        self,
        images_dir: str | Path,
        model_id: str = "LiquidAI/LFM2-VL-450M",
        prompt: str = "What is in this image?",
        max_samples: Optional[int] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.model_id = model_id
        self.prompt = prompt
        self.max_samples = max_samples

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Calibration images directory not found: {self.images_dir}")

        # Load processor once
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images: List[Path] = [p for p in sorted(self.images_dir.iterdir()) if p.suffix.lower() in exts]
        if self.max_samples is not None:
            all_images = all_images[: self.max_samples]
        if not all_images:
            raise RuntimeError(f"No images found in {self.images_dir}")
        self._image_paths = all_images

        # Prepare an iterator lazily
        self._enum_data_dicts: Optional[Iterator[Dict[str, np.ndarray]]] = None

    def _iter_batches(self) -> Iterable[Dict[str, np.ndarray]]:
        for img_path in self._image_paths:
            image = Image.open(img_path).convert("RGB")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            features = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            sample: Dict[str, np.ndarray] = {}
            for name in ONNX_INPUT_NAMES:
                if name not in features:
                    # Some inputs might be optional depending on export settings
                    continue
                tensor = features[name]
                if name in {"input_ids", "attention_mask", "pixel_attention_mask", "spatial_shapes"}:
                    sample[name] = tensor.numpy().astype(np.int64)
                else:
                    sample[name] = tensor.numpy().astype(np.float32)

            # Only yield non-empty dicts
            if sample:
                yield sample

    def get_next(self) -> Optional[Dict[str, np.ndarray]]: 
        if self._enum_data_dicts is None:
            self._enum_data_dicts = iter(self._iter_batches())
        return next(self._enum_data_dicts, None)

    def rewind(self) -> None: 
        #Reset the iterator so quantizer can make multiple passes if needed.
        self._enum_data_dicts = iter(self._iter_batches())
