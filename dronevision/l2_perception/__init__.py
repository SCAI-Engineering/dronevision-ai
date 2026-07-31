"""Layer 2 — perception: find the drone in a single camera image.

    detect(image_rgb, cam=None) -> (u, v) pixel position, or None

Every detector satisfies that one signature, so the backend is swappable without
touching triangulation or control. Backends fall into two groups:

  * neural — a small YOLO detecting the drone by shape. This is the layer that
    dominates cost on an Arm CPU, and therefore the layer the optimization work
    targets: inference runtime, quantization, and input resolution.
  * classical — colour-marker and motion (background-subtraction) detectors.
    Far cheaper, and useful as reference points for how much the network
    actually buys.

Which *inference runtime* executes the network is a separate axis from which
detector is chosen, because the same weights run very differently across Arm
cores — dot-product instructions exist on some and not others.
"""
