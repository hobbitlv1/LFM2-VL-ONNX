# LFM2-VL-ONNX

A toolkit for quantizing, evaluating, and benchmarking ONNX models, specifically designed for model LiquidAI/LFM2-VL-450M.

## Overview

This repository provides tools to export models to ONNX format, apply dynamic quantization to reduce model size, evaluate quantization accuracy on COCO validation dataset, benchmark inference performance, and prepare validation datasets for consistent evaluation.

## Installation

### Linux

```bash
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install optimum[onnxruntime] transformers pillow numpy torch tqdm

# For GPU support
pip install optimum[onnxruntime-gpu] transformers pillow numpy torch tqdm

# pip will install old version of transformers, to avoid errors update using this command
pip install --upgrade transformers

# To download dataset use these commands
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### macOS

```bash
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install optimum[onnxruntime] transformers pillow numpy torch tqdm

# pip will install old version of transformers, to avoid errors update using this command
pip install --upgrade transformers

# To download dataset use these commands
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Windows

```powershell
# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install optimum[onnxruntime] transformers pillow numpy torch tqdm

# For GPU support
pip install optimum[onnxruntime-gpu] transformers pillow numpy torch tqdm

# pip will install old version of transformers, to avoid errors update using this command
pip install --upgrade transformers

# To download dataset
Go to https://cocodataset.org/ and download 2017 Val images [5K/1GB] and 2017 Train/Val annotations [241MB]
```

## Project Structure

```

exporter.py                 # Export models to ONNX format
dynamically_quantize.py     # Apply dynamic quantization
evaluate_models.py          # Compare original vs quantized models
prepare_validation_data.py  # Prepare COCO validation dataset
speed_check.py              # Benchmark model performance
debug_quantization.py       # Debug and compare model outputs
```

## Usage

For detailed arguments on any script, run `python scriptname.py -h` to see all available options.

### 1. Export Model to ONNX

Export your model to ONNX format using the exporter script:

```bash
# Linux/macOS
python exporter.py --use-cache --run-symbolic-shape-inference

# Windows
python exporter.py --use-cache --run-symbolic-shape-inference
```

### 2. Prepare Validation Dataset

Download COCO validation images and annotations from the official COCO dataset website, then prepare the validation dataset. The script expects COCO validation images in a `val2017` directory and annotations in `annotations/captions_val2017.json`.

```bash
# Linux/macOS
python prepare_validation_data.py --annotations annotations/captions_val2017.json --images-dir val2017 --output quant/validation_dataset.pt --limit 10

# Windows
python prepare_validation_data.py --annotations annotations\captions_val2017.json --images-dir val2017 --output quant\validation_dataset.pt --limit 10
```

### 3. Apply Dynamic Quantization

Quantize the exported ONNX model to reduce size and improve performance. The script loads the preprocessed ONNX model from `onnx_optimum/preprocessed/model.pre.ort.onnx`, applies dynamic quantization with QUInt8 weights, and saves the quantized model to `quant/model_dynamic.quant.onnx`.

```bash
# All platforms
python dynamically_quantize.py
```

### 4. Evaluate Model Accuracy

Compare the original and quantized models on validation data to measure accuracy degradation from quantization:

```bash
# Linux/macOS
python evaluate_models.py --original onnx_optimum/model.onnx --quantized quant/model_dynamic.quant.onnx --num-samples 10 --verbose

# Windows
python evaluate_models.py --original onnx_optimum\model.onnx --quantized quant\model_dynamic.quant.onnx --num-samples 10 --verbose
```

The evaluation provides argmax accuracy (percentage of matching top predictions), top-5 overlap metrics, distribution similarity measures like KL divergence and cosine similarity, and per-sample analysis of mismatched predictions.

The model performance varies during evaluation. The random seed is set to 420 which will produce a result of 8/10, you can experiment to get different results.

### 5. Benchmark Performance

Measure inference latency and throughput to compare performance between original and quantized models:

```bash
# Linux/macOS
python speed_check.py --reference onnx_optimum/model.onnx --candidate quant/model_dynamic.quant.onnx --runs 50 --warmup 10

# Windows
python speed_check.py --reference onnx_optimum\model.onnx --candidate quant\model_dynamic.quant.onnx --runs 50 --warmup 10
```

The benchmark reports average latency, percentile latencies (P50, P90, P95, P99), throughput in inferences per second, and relative speedup compared to the baseline model.

### 6. Debug Quantization Output

For detailed debugging and comparison of model outputs on a single test image, use the debug quantization script. This is useful for investigating quantization issues or verifying model behavior:

```bash
# All platforms
python debug_quantization.py
```

The script loads a test image from `val2017/000000000139.jpg` and compares the logits between original and quantized models. It reports logit statistics (min, max, mean, std), top-5 predictions with token IDs and scores, absolute and relative differences between models, and warnings for potential issues like NaN values or unusually large logits. This helps identify if quantization is causing unexpected behavior in model outputs.

## Model Support

Primary support for LiquidAI/LFM2-VL-450M vision-language model. The toolkit can be adapted for other ONNX-compatible models with appropriate modifications to input preprocessing.

## Performance Tips

Use GPU providers (CUDA) when available for faster inference.
Change settings
