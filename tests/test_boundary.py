"""Guards the rule the whole project rests on: this repository is the AI block.

The drone, the cameras and the control software belong to the simulator project and
are consumed as network services. Nothing here may link against them. If that
slips, the AI stops being deployable to an Arm device and stops being benchmarkable
without a simulator — and it slips silently, because on a development machine that
has everything installed, the import just works.

Checked by parsing the source rather than importing, so a forbidden dependency is
caught even when it happens to be available locally.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dronevision"

# Each of these belongs to a different block of the system.
FORBIDDEN = {
    "gz": "Gazebo — the camera service publishes frames; consume them over the network",
    "pymavlink": "flight stack — the control software consumes our estimate",
    "mavsdk": "flight stack — the control software consumes our estimate",
    "rospy": "robot middleware — not part of the AI block",
    "rclpy": "robot middleware — not part of the AI block",
}

# Heavy inference dependencies: allowed, but only inside a function body, so that
# importing the package never drags a runtime in. A Pi may have exactly one.
LAZY_ONLY = {"torch", "ultralytics", "onnxruntime", "executorch"}


def modules(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def imports(tree):
    """(root_module, node, is_module_scope) for every import in the tree."""
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def module_scope(node):
        p = parent.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            p = parent.get(p)
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node, module_scope(node)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split(".")[0], node, module_scope(node)


def test_package_is_not_empty():
    """Stops the guard passing vacuously over a moved or renamed directory."""
    assert len(modules(PACKAGE)) >= 12


@pytest.mark.parametrize("path", modules(PACKAGE), ids=lambda p: p.name)
def test_no_foreign_block_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for root, node, _ in imports(tree):
        if root in FORBIDDEN:
            pytest.fail(f"{path.relative_to(ROOT)}:{node.lineno} imports {root!r}\n"
                        f"  {FORBIDDEN[root]}")


@pytest.mark.parametrize("path", modules(PACKAGE), ids=lambda p: p.name)
def test_inference_runtimes_are_imported_lazily(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for root, node, module_scope in imports(tree):
        if root in LAZY_ONLY and module_scope:
            pytest.fail(
                f"{path.relative_to(ROOT)}:{node.lineno} imports {root!r} at module "
                f"scope.\n  Move it inside the function that needs it: a device with "
                f"one runtime installed must still import this package.")


def test_the_repository_holds_no_other_block():
    """No control software, no simulator bridge, no world files."""
    strays = [p for p in ROOT.rglob("*.py")
              if "__pycache__" not in p.parts
              and p.parts[len(ROOT.parts)] in ("control", "bridge", "simulation",
                                               "worlds", "px4")]
    assert not strays, (
        f"these belong to the simulator project, not the AI block: "
        f"{[str(p.relative_to(ROOT)) for p in strays]}")


def test_importing_the_pipeline_pulls_in_nothing_foreign():
    """The static checks prove the source is clean; this proves the effect."""
    import dronevision                                    # noqa: F401
    import dronevision.l3_association.associator             # noqa: F401
    import dronevision.l3_association.tracker                # noqa: F401
    import dronevision.l5_estimation.ekf3d                   # noqa: F401
    import dronevision.l3_association.kalman2d                # noqa: F401
    import dronevision.l5_estimation.smoothing               # noqa: F401
    import dronevision.io                             # noqa: F401
    import dronevision.l2_perception.detector                # noqa: F401
    import dronevision.pipeline                           # noqa: F401
    import dronevision.service                            # noqa: F401
    import dronevision.l1_image_sync.aligner                       # noqa: F401
    import dronevision.io.sources.replay                # noqa: F401
    import dronevision.l4_triangulation.geometry             # noqa: F401

    leaked = [m for m in list(FORBIDDEN) + sorted(LAZY_ONLY) if m in sys.modules]
    assert not leaked, (
        f"importing the AI pulled in {leaked}; it must install and run on a device "
        f"that has none of them")


def test_the_two_service_contracts_are_documented_here():
    """The AI owns both wire formats, so both must be described in this repo —
    that is what lets either service be reimplemented against it."""
    net = (PACKAGE / "io" / "sources" / "net.py").read_text(encoding="utf-8")
    schema = (PACKAGE / "io" / "schema.py").read_text(encoding="utf-8")
    assert "truth" in net and "CONFLATE" in net
    assert "src_t" in schema and "pos_enu" in schema


# --------------------------------------------------------------------------
# The structure IS the architecture diagram
# --------------------------------------------------------------------------

def test_the_five_layers_exist_in_diagram_order():
    """`dronevision.LAYERS` is the diagram, as data. The directories must match it.

    Locked down because the ordinal prefixes are the only thing making processing
    order visible in a directory listing, and a rename that breaks the mapping
    leaves the code silently disagreeing with the architecture it documents.
    """
    from dronevision import LAYERS

    names = [n for n, _ in LAYERS]
    assert names == sorted(names), "prefixes must sort into processing order"
    assert len(names) == 5

    for i, (name, label) in enumerate(LAYERS, start=1):
        d = PACKAGE / name
        assert d.is_dir(), f"layer {i} ({label}) has no directory {name}/"
        assert (d / "__init__.py").exists(), f"{name}/ is not a package"
        assert name.startswith(f"l{i}_"), f"{name} is not prefixed l{i}_"
        head = (d / "__init__.py").read_text(encoding="utf-8")[:400]
        assert f"Layer {i}" in head, (
            f"{name}/__init__.py must open by naming itself 'Layer {i}' — the "
            f"docstring is how a reader maps code to the diagram")


def test_nothing_masquerades_as_a_layer():
    """Only the five layers carry an l<n>_ prefix, and io/ is the only other
    subpackage. A sixth 'layer' means the diagram needs updating first."""
    from dronevision import LAYERS

    subpackages = {d.name for d in PACKAGE.iterdir()
                   if d.is_dir() and (d / "__init__.py").exists()}
    expected = {n for n, _ in LAYERS} | {"io"}
    assert subpackages == expected, (
        f"unexpected subpackages: {sorted(subpackages - expected)}; "
        f"missing: {sorted(expected - subpackages)}")


def test_layers_do_not_reach_sideways_into_later_layers():
    """Data flows one way. A layer importing a *later* layer means the pipeline
    order in the diagram is not the order the code actually runs in."""
    import ast

    from dronevision import LAYERS

    order = {name: i for i, (name, _) in enumerate(LAYERS)}
    offenders = []
    for name, _ in LAYERS:
        for path in modules(PACKAGE / name):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                mod = (node.module if isinstance(node, ast.ImportFrom) else None)
                if not mod:
                    continue
                for other, j in order.items():
                    if f"dronevision.{other}" in mod and j > order[name]:
                        offenders.append(f"{path.name} -> {other}")
    assert not offenders, f"backwards layer dependencies: {offenders}"
