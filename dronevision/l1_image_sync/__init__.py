"""Layer 1 — Image Synchronization.

Turns independently-arriving camera frames into sets the rest of the pipeline can
reason about, and decides which camera the detector looks at next.

The cameras do not publish in lockstep: measured against a live service, frames in
one "set" sit up to 84 ms apart, mean 39 ms. Triangulating across views captured at
different instants places the target where it never was — negligible while
hovering, and proportional to speed otherwise.

This layer also owns the *scheduling* decision. On a CPU-bound Arm device that
choice dominates throughput: running the network on every camera every tick is what
makes the naive pipeline too slow to close a control loop.

Frame acquisition itself is not here. That is a boundary, and boundaries live in
`dronevision.io.sources`.
"""
