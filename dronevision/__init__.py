"""dronevision — the AI block: multi-camera 3D localization for Arm edge devices.

Locates a flying drone in 3D from several fixed cameras, with no GPS and no onboard
sensing, fast enough to close a control loop on a Raspberry Pi.

THE FIVE LAYERS, in processing order. The directory names carry their ordinal so a
listing reads top-to-bottom like the architecture diagram, instead of
alphabetically:

    l1_image_sync/      1. Image Synchronization
    l2_perception/      2. Perception Layer (YOLO)
    l3_association/     3. 2D Detection Association
    l4_triangulation/   4. DLT Triangulation
    l5_estimation/      5. EKF State Estimation and Filtering

Everything else is deliberately not a layer:

    io/                 the boundaries — frames in, state estimate out
    pipeline.py         chains the five layers
    service.py          runs the block as a process

SCOPE. This project is the AI block and nothing else. The drone, the cameras and
the control software belong to the simulator project and are consumed as network
services: frames arrive over ZeroMQ, the position estimate leaves over UDP. So
nothing here imports Gazebo, ROS, PX4 or MAVLink — which is what lets this install
and run on a bare Raspberry Pi, and lets the benchmarks run with no simulator
present at all. `tests/test_boundary.py` enforces it.
"""

__version__ = "0.1.0"

#: The five layers, in processing order. Kept as data so tooling and documentation
#: cannot drift from the diagram.
LAYERS = (
    ("l1_image_sync", "Image Synchronization"),
    ("l2_perception", "Perception Layer (YOLO)"),
    ("l3_association", "2D Detection Association"),
    ("l4_triangulation", "DLT Triangulation"),
    ("l5_estimation", "EKF State Estimation and Filtering"),
)
