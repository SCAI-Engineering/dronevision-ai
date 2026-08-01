"""ONNX Runtime backend — the portable one, and the first Arm target.

Chosen to lead because it has prebuilt aarch64 wheels for both boards, so the same
artifact and the same code measure a Pi 4 and a Pi 5 with no build step in between.

The session options here are not incidental. `intra_op_num_threads` fixes the unit of
work; `CPUExecutionProvider` is named explicitly so a provider that happens to be
installed cannot quietly change what a row in the results table means; and spin-waiting
is disabled because ORT's default busy-wait between inferences keeps cores hot, which on a
passively cooled Pi 4 shows up as thermal throttling partway through a sweep and reads as
a performance regression that is really a temperature problem.
"""
from dronevision.l2_perception.runtimes.base import TensorRuntime, TensorSpec

_ORT_TO_NP = {
    "tensor(float)": "float32", "tensor(float16)": "float16",
    "tensor(uint8)": "uint8", "tensor(int8)": "int8",
}


class OnnxRuntime(TensorRuntime):
    """Executes a `.onnx` graph on CPU."""

    name = "onnx"

    def __init__(self, model, imgsz=None, threads=None, nc=1,
                 providers=None, allow_spinning=False, **kw):
        super().__init__(model, imgsz=imgsz, threads=threads, nc=nc, **kw)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                f"ONNX Runtime is not installed ({e}).\n"
                f"  pip install -e \".[ort]\"") from None

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads.intra
        so.inter_op_num_threads = self.threads.inter
        # Sequential execution means inter_op is inert; it is still reported so a results
        # row never implies a knob was doing something it was not.
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not allow_spinning:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.model_path, sess_options=so,
                                         providers=self.providers)
        self._in = self.sess.get_inputs()[0]
        self._outs = self.sess.get_outputs()
        self._out_names = [o.name for o in self._outs]
        self._meta = dict(self.sess.get_modelmeta().custom_metadata_map or {})
        self._check_metadata()

    def _check_metadata(self):
        """Cross-check the graph against what the exporter recorded.

        Sniffing the layout from shapes is a fallback; when the exporter left metadata,
        the two must agree. A disagreement means the file is not what it claims, and
        continuing would produce numbers attributed to the wrong artifact.
        """
        shape = self._in.shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            graph_size = int(shape[2])
            if self.imgsz and int(self.imgsz) != graph_size:
                raise ValueError(
                    f"{self.model_path}: graph input is {graph_size}x{graph_size} but "
                    f"imgsz={self.imgsz} was requested. The graph is static — re-export "
                    f"at the size you want rather than resizing around it, or the "
                    f"comparison measures different amounts of work.")
            self.imgsz = graph_size

        declared = self._meta.get("imgsz")
        if declared and self.imgsz:
            try:
                want = int(str(declared).strip("[]").split(",")[0])
                if want != int(self.imgsz):
                    raise ValueError(
                        f"{self.model_path}: metadata says imgsz {want} but the graph "
                        f"input is {self.imgsz}")
            except (ValueError, IndexError) as e:
                if "metadata says" in str(e):
                    raise

    # -- TensorRuntime contract ---------------------------------------------

    def infer(self, x_nchw):
        return self.sess.run(self._out_names, {self._in.name: x_nchw})

    @property
    def input_spec(self):
        return TensorSpec(self._in.name, tuple(self._in.shape),
                          _ORT_TO_NP.get(self._in.type, self._in.type))

    @property
    def output_specs(self):
        return tuple(TensorSpec(o.name, tuple(o.shape),
                                _ORT_TO_NP.get(o.type, o.type)) for o in self._outs)

    def describe(self):
        d = super().describe()
        import onnxruntime as ort
        d.update({
            "ort_version": ort.__version__,
            "providers": self.sess.get_providers(),
            "end2end": self._meta.get("end2end"),
            "exporter": self._meta.get("version"),
            "precision": self._infer_precision(),
        })
        return d

    def _infer_precision(self):
        """Best-effort label for the table: what the weights actually are."""
        name = self.model_path.lower()
        for tag in ("int8", "uint8", "fp16", "half", "qdq"):
            if tag in name:
                return "int8" if tag in ("int8", "uint8", "qdq") else "fp16"
        if self.input_spec.dtype in ("uint8", "int8"):
            return "int8"
        return "fp32"

    def close(self):
        self.sess = None
