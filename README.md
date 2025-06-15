# ML_tensorflowLite
This repo explains on how to use tensorflowlite


Explaining Post-Training Quantization
Post-Training Quantization (PTQ) in TensorFlow (specifically for TensorFlow Lite) is an optimization technique that you apply after your machine learning model has been fully trained using standard 32-bit floating-point precision. The goal is to convert the model's weights and activations to lower-precision formats (like 8-bit integers or 16-bit floats) to reduce model size and improve inference speed, particularly for deployment on edge devices.

Here's a detailed breakdown of how PTQ works with TensorFlow:

The Core Idea
Instead of modifying the training process (as in Quantization-Aware Training), PTQ takes an already trained model and, during its conversion to the TensorFlow Lite format (.tflite), it applies the quantization logic. This makes it a very convenient and often sufficient approach for many use cases, as it doesn't require retraining or fine-tuning.

How it Works (Under the Hood)
When you convert a TensorFlow model to TensorFlow Lite with PTQ enabled, the TensorFlow Lite Converter (specifically tf.lite.TFLiteConverter) performs the following:

Analyze Tensor Ranges: For floating-point tensors (like weights, biases, and intermediate activations), the converter needs to determine their min and max values. This range is crucial for mapping the floating-point values to a fixed-point integer range.

For weights and biases: These are static and their ranges can be directly calculated from the trained model.
For activations (dynamic tensors): The ranges of intermediate activations are dynamic and depend on the input data. This is where the different PTQ types come into play.
Apply Quantization Scales and Zero Points: Based on the determined ranges, the converter calculates a scale and zero_point for each tensor. These parameters are embedded in the .tflite model and are used during inference to convert between floating-point and integer representations:
integer_value=round(real_value/scale+zero_point)
real_value=(integer_value−zero_point)×scale

Convert to Lower Precision: The actual values (weights and, depending on the type, activations) are then converted to the chosen lower-precision format (e.g., 8-bit integer, 16-bit float).
