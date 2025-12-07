import tensorflow as tf
import tf2onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- CONFIGURATION ---
H5_PATH = 'mobilenetv2_roboflow_classification (6).h5'
ONNX_FP32_PATH = 'Hand_Gestures_model_fp32.onnx'
ONNX_PI_PATH = 'Hand_Gestures_model_pi_quant.onnx'

try:
    print(f"Loading Keras model from {H5_PATH}...")
    model = tf.keras.models.load_model(H5_PATH)

    # 1. CONVERT TO FP32 ONNX
    # Opset 13 is best for Raspberry Pi
    print("Converting to standard ONNX (FP32)...")
    spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=ONNX_FP32_PATH
    )
    print(f"Standard model saved to: {ONNX_FP32_PATH}")

    # 2. QUANTIZE FOR RASPBERRY PI (INT8)
    # This reduces size by 4x and improves CPU inference speed
    print("\nQuantizing model for Raspberry Pi (Int8)...")

    quantize_dynamic(
        model_input=ONNX_FP32_PATH,
        model_output=ONNX_PI_PATH,
        weight_type=QuantType.QUInt8
    )

    print(f"SUCCESS! Optimized model saved to: {ONNX_PI_PATH}")
    print(f"Transfer '{ONNX_PI_PATH}' to your Raspberry Pi.")

except Exception as e:
    print(f"\nERROR: {e}")