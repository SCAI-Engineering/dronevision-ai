#!/usr/bin/env python3
"""Refine Projection Matrices via Bundle Adjustment (Reprojection Error Minimization).

This tool takes a recorded corpus and an existing site calibration, then optimizes 
the camera projection matrices to minimize the reprojection error against ground truth.

It treats the current calibration as a starting point and performs a non-linear 
least-squares optimization on the camera extrinsics (translation and rotation).

    python -m bench.refine_calibration --site factory --detector color
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize

from dronevision.l4_triangulation.calibration import load_site, Calibration, Camera
from dronevision.l4_triangulation.geometry import reproj_err
from dronevision.l2_perception.detector import make_detector
from dronevision.io.sources.replay import ReplaySource

def rpy_to_R(roll, pitch, yaw):
    """Body->world rotation from roll/pitch/yaw."""
    import math
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    return Rz @ Ry @ Rx

def build_P(K, pos, rpy, axes_conv):
    """Reconstruct P from K, position and RPY."""
    # World->camera rotation
    R = axes_conv @ rpy_to_R(*rpy).T
    # t = -R @ position
    t = (-R @ np.asarray(pos, float)).reshape(3, 1)
    P = K @ np.hstack([R, t])
    return P

def cost_function(params, data, site_info):
    """Total squared reprojection error across all cameras and frames."""
    # params: [cam0_tx, cam0_ty, cam0_tz, cam0_rx, cam0_ry, cam0_rz, ...]
    num_cams = len(site_info['cams'])
    current_params = params.reshape((num_cams, 6))
    
    total_err = 0.0
    for i, cam_name in enumerate(site_info['cams']):
        # Update parameters for this camera
        p = current_params[i]
        pos = np.asarray(site_info['init_pos'][cam_name]) + p[:3]
        rpy = np.asarray(site_info['init_rpy'][cam_name]) + p[3:]
        
        P = build_P(site_info['K'][cam_name], pos, rpy, site_info['axes'])
        
        # Calculate error for all frames where this camera has a detection
        for truth, uv in data[cam_name]:
            h = P @ np.append(truth, 1.0)
            if h[2] <= 1e-9:
                total_err += 1e6 # Penalty for point behind camera
                continue
            proj = h[:2] / h[2]
            total_err += np.sum((proj - uv)**2)
            
    return total_err

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--site", default="factory")
    ap.add_argument("--detector", default="color")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--marker-offset", action="store_true",
                    help="account for detector marker offset in reprojection (subtract target_dz from truth z)")
    a = ap.parse_args(argv)

    site = load_site(a.site)
    det_kw = {"runtime": a.runtime} if a.detector in ("yolo", "hybrid") else {}
    detector = make_detector(a.detector, **det_kw)
    src = ReplaySource(a.corpus)

    print(f"Refining calibration for site: {site.name}")
    
    # Determine offset to apply to truth for reprojection
    # The detector sees the marker at +marker_dz above vehicle origin.
    # For reprojection error minimization, we want to project the detected point,
    # so we need truth shifted up by marker_dz to match what detector sees.
    if a.marker_offset:
        offset_key = getattr(detector, "target_offset_key", None)
        target_dz = site.target_offset(offset_key) if offset_key else site.marker_dz
        print(f"Using marker offset correction: {target_dz:.4f} m (truth shifted up)")
    else:
        target_dz = 0.0
        print("No marker offset correction (truth used as-is)")
    
    # 1. Collect data: {cam: [(truth, uv), ...]}
    data = {name: [] for name in site.cam_names}
    for s in src:
        truth = s.truth
        if truth is None: continue
        # Shift truth up by marker offset so reprojection matches detected point
        truth_shifted = np.asarray(truth) + np.array([0, 0, target_dz])
        for cam in site.cam_names:
            img = s.latest(cam)
            if img is None: continue
            uv = detector.detect(img, cam=cam)
            if uv:
                data[cam].append((truth_shifted, np.asarray(uv)))

    # 2. Prepare optimization state
    site_info = {
        'cams': site.cam_names,
        'init_pos': {n: c.position for n, c in site.cameras.items()},
        'init_rpy': {n: c.rpy for n, c in site.cameras.items()},
        'K': {n: c.K for n, c in site.cameras.items()},
        'axes': site.cameras[site.cam_names[0]].R @ np.linalg.inv(rpy_to_R(*site.cameras[site.cam_names[0]].rpy).T) 
        # This is a bit hacky to get the AXIS_CONVENTIONS matrix from an existing camera
    }
    # Correcting axes: just use the one from calibration.py if possible, but we can't import it easily.
    # Let's assume 'optical' or extract it. Actually, let's just use a simpler approach for P.
    # Since we want to refine P, and P = K[R|t], we can optimize R and t directly.
    
    # Initial guess: all zeros (no change)
    initial_guess = np.zeros(len(site.cam_names) * 6)

    print("Optimizing Projection Matrices... this may take a minute...")
    res = minimize(cost_function, initial_guess, args=(data, site_info), method='L-BFGS-B', options={'maxiter': 1000})

    if not res.success:
        print(f"Optimization failed: {res.message}")
        return 1

    # 3. Report results
    refined_params = res.x.reshape((len(site.cam_names), 6))
    print("\n--- Refinement Results ---")
    print(f"{'Camera':<12} {'Delta Pos (m)':>20} {'Delta RPY (rad)':>20}")
    print("-" * 60)
    for i, name in enumerate(site.cam_names):
        dp = refined_params[i, :3]
        dr = refined_params[i, 3:]
        print(f"{name:<12} {str(np.round(dp, 4)):>20} {str(np.round(dr, 4)):>20}")

    # 4. Suggest YAML updates
    print("\nSuggested config updates for `config/%s.yaml`:" % a.site)
    for i, name in enumerate(site.cam_names):
        p = refined_params[i]
        new_pos = site.cameras[name].position + p[:3]
        new_rpy = np.asarray(site.cameras[name].rpy) + p[3:]
        print(f"\n# {name}")
        print(f"  position: {list(np.round(new_pos, 6))}")
        print(f"  rpy: {list(np.round(new_rpy, 6))}")

    src.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
