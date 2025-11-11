import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'onnx_optimum/preprocessed/model.pre.ort.onnx'
model_quant = 'quant/model_dynamic.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8, per_channel=True)
