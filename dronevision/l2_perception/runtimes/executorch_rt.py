"""ExecuTorch backend — placeholder.

This is the runtime with the strongest claim on an Arm submission: on CPUs that carry
`FEAT_DotProd` it dispatches int8 matmuls to KleidiAI's SDOT micro-kernels through
XNNPACK, which is the Arm-specific acceleration path rather than a generic one.

It is deliberately unimplemented for now. The Pi 4 in hand is a Cortex-A72 (ARMv8.0) whose
`/proc/cpuinfo` reports `fp asimd evtstrm crc32 cpuid` and no `asimddp`, so those kernels
would not be selected on it at all — the gain only appears on the Cortex-A76 in a Pi 5.
Standing this up before there is a board that can show the difference would produce a row
in the results table that measures nothing in particular.

To implement: export a `.pte` at the same imgsz as every other artifact, load it via
`executorch.runtime`, and subclass `TensorRuntime` supplying `infer()`, `input_spec` and
`output_specs`. The existing codec handles the rest, including the raw output layout a
quantized export is likely to have.
"""
from dronevision.l2_perception.runtimes.base import TensorRuntime


class ExecuTorchRuntime(TensorRuntime):
    name = "executorch"

    def __init__(self, *a, **kw):
        raise NotImplementedError(
            "The ExecuTorch runtime is not implemented yet.\n"
            "It is the KleidiAI path and only pays off on a core with FEAT_DotProd "
            "(Cortex-A76 / Pi 5); the Pi 4's Cortex-A72 lacks it.\n"
            "Use --runtime onnx in the meantime.")

    def infer(self, x_nchw):
        raise NotImplementedError
