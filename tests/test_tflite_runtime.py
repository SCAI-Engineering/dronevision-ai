import numpy as np

from dronevision.l2_perception.runtimes.tflite_rt import TfLiteRuntime


def test_prepare_input_uses_exact_uint8_to_int8_mapping():
    runtime = object.__new__(TfLiteRuntime)
    runtime.in_shape = (1, 3, 1, 3)
    runtime.in_dtype = np.dtype(np.int8)
    runtime.in_quant = (1.0 / 255.0, -128)
    rgb = np.array([[[0, 127, 255], [1, 128, 254], [255, 0, 64]]], dtype=np.uint8)

    actual = runtime._prepare_input(rgb)
    expected = rgb.transpose(2, 0, 1)[None].astype(np.int16) - 128

    np.testing.assert_array_equal(actual, expected.astype(np.int8))
