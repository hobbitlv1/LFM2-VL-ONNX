import argparse
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
from lfm2vl_data_reader import Lfm2VlCalibrationDataReader
import gc


def main():
    parser = argparse.ArgumentParser(description='Perform static quantization on ONNX models')

    # Model paths
    parser.add_argument('--input', '-i', type=str,
                        default='onnx_optimum/preprocessed/model.pre.ort.onnx',
                        help='Path to input FP32 ONNX model (default: %(default)s)')
    parser.add_argument('--output', '-o', type=str,
                        default='quant/model_static.quant.onnx',
                        help='Path to output quantized model (default: %(default)s)')

    # Calibration configuration
    parser.add_argument('--calibration-dir', type=str,
                        default='val2017',
                        help='Directory containing calibration images (default: %(default)s)')
    parser.add_argument('--calibration-samples', type=int,
                        default=50,
                        help='Number of calibration samples to use (default: %(default)s)')
    parser.add_argument('--prompt', type=str,
                        default='What is in this image?',
                        help='Prompt to use for image processing (default: %(default)s)')

    # Quantization parameters
    parser.add_argument('--quant-format', type=str,
                        default='QDQ',
                        choices=['QOperator', 'QDQ'],
                        help='Quantization format (default: %(default)s)')
    parser.add_argument('--activation-type', type=str,
                        default='QUInt8',
                        choices=['QInt8', 'QUInt8'],
                        help='Activation quantization type (default: %(default)s)')
    parser.add_argument('--weight-type', type=str,
                        default='QUInt8',
                        choices=['QInt8', 'QUInt8'],
                        help='Weight quantization type (default: %(default)s)')
    parser.add_argument('--calibrate-method', type=str,
                        default='MinMax',
                        choices=['MinMax', 'Entropy', 'Percentile'],
                        help='Calibration method (default: %(default)s)')
    parser.add_argument('--op-types', type=str,
                        nargs='+',
                        default=['MatMul', 'Conv'],
                        help='Operation types to quantize (default: %(default)s)')
    parser.add_argument('--per-channel', action='store_true',
                        default=True,
                        help='Use per-channel quantization for weights (default: True)')
    parser.add_argument('--no-per-channel', dest='per_channel', action='store_false',
                        help='Disable per-channel quantization')
    parser.add_argument('--reduce-range', action='store_true',
                        default=False,
                        help='Reduce quantization range (default: False)')

    args = parser.parse_args()

    quant_format_map = {
        'QOperator': QuantFormat.QOperator,
        'QDQ': QuantFormat.QDQ
    }
    quant_type_map = {
        'QInt8': QuantType.QInt8,
        'QUInt8': QuantType.QUInt8
    }
    calibrate_method_map = {
        'MinMax': CalibrationMethod.MinMax,
        'Entropy': CalibrationMethod.Entropy,
        'Percentile': CalibrationMethod.Percentile
    }

    # Create calibration data reader
    print(f"\nLoading calibration data from: {args.calibration_dir}...")
    data_reader = Lfm2VlCalibrationDataReader(
        images_dir=args.calibration_dir,
        model_id="LiquidAI/LFM2-VL-450M",
        prompt=args.prompt,
        max_samples=args.calibration_samples
    )

    gc.collect()

    # Only quantize MatMul and Conv

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Op types: {args.op_types}")

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=data_reader,
        quant_format=quant_format_map[args.quant_format],
        activation_type=quant_type_map[args.activation_type],
        weight_type=quant_type_map[args.weight_type],
        calibrate_method=calibrate_method_map[args.calibrate_method],
        op_types_to_quantize=args.op_types,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        use_external_data_format=False,
    )

    print(f"\nQuantized model saved to: {args.output}")
    print(f"Calibration used {args.calibration_samples} samples")

    gc.collect()


if __name__ == '__main__':
    main()
