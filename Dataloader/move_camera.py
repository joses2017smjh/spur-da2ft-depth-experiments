"""
move_camera.py — Generate camera poses along a semi-circular path around
the tree trunk.  For each angular position on the arc, produces one pose
per Z-level (same vertical sweep as the original center shots).

The initial camera position defines the **center** of the arc.  The arc
spans CENTRAL_ANGLE (default pi = semi-circle) symmetrically around the
reference viewpoint: half the angle to the left, half to the right.

Camera rotation uses ROTATION_REF as the base, adjusted by the angular
offset around the trunk (rz += delta_theta).

Usage (standalone test):
    python move_camera.py

When imported from generate_tree2:
    from move_camera import generate_semicircle_poses
    poses = generate_semicircle_poses(cylinder_json_path, x_ref, y_ref,
                                      z_ref, z_final, rotation_ref=ROTATION_REF,
                                      trunk_part_name="trunk_1",
                                      num_poses=6, num_z_levels=6)
"""

import json
import math
import random
import numpy as np

# Number of angular positions along the arc
NUM_POSES = 6

# Number of Z-levels per camera position
NUM_Z_LEVELS = 6

# Central angle of the arc (radians).  pi = semi-circle.
# The arc is centered on the reference camera: half the angle left, half right.
CENTRAL_ANGLE = np.pi

# Random Y offset range: positive = farther from tree (camera Y > tree Y)
Y_OFFSET_MIN = 0.0
Y_OFFSET_MAX = 0.35

# Default cylinder JSON (bark_brown, first envy tree)
DEFAULT_CYLINDER_JSON = (
    "/nfs/stak/users/sanchej7/hpc-share/Computer_Vision"
    "/Data/full_trunk/cylinders_world/bark_brown/lpy_envy_00000.json"
)

# Fallback reference values (same as generate_tree2.py)
_X_REF = -9.820750732421875
_Y_REF = -6.95839786529541
_Z_REF = 0.85
_Z_FINAL = 3.73232572555542
_ROTATION_REF = [1.5395008325576782, 0.0, 3.194649314880371]


# ── Trunk axis from cylinder data ────────────────────────────────────────────

def load_trunk_axis(cylinder_json_path, trunk_part_name="trunk_1"):
    """
    Load cylinder JSON, filter by *trunk_part_name*, and return (P_base, P_top)
    as numpy arrays — the bottom and top centroids of the trunk axis.
    """
    with open(cylinder_json_path, "r") as f:
        cylinders = json.load(f)

    trunk_centroids = [
        np.array(c["centroid"])
        for c in cylinders
        if c.get("part_name") == trunk_part_name
    ]
    if not trunk_centroids:
        raise ValueError(
            f"No '{trunk_part_name}' cylinders found in {cylinder_json_path}"
        )

    trunk_centroids.sort(key=lambda p: p[2])
    p_base = trunk_centroids[0]
    p_top = trunk_centroids[-1]
    return p_base, p_top


# ── Main pose generator ─────────────────────────────────────────────────────

def generate_semicircle_poses(cylinder_json_path=None, x_ref=None, y_ref=None,
                              z_ref=None, z_final=None, rotation_ref=None,
                              trunk_part_name="trunk_1",
                              num_poses=NUM_POSES, num_z_levels=NUM_Z_LEVELS,
                              central_angle=CENTRAL_ANGLE, seed=None):
    """
    Generate camera poses on an arc around the trunk.

    The reference camera position (x_ref, y_ref) defines the **center** of
    the arc.  The arc spans *central_angle* radians symmetrically around
    this position (half left, half right).

    For each of *num_poses* angular positions, generates *num_z_levels*
    poses at different heights (Z linearly spaced from z_ref to z_final).

    Parameters
    ----------
    cylinder_json_path : str
        Path to the cylinders_world JSON for the tree.
    x_ref, y_ref, z_ref, z_final : float
        Reference camera position and final Z height.
    rotation_ref : list of float
        Reference camera rotation [rx, ry, rz] (Euler XYZ, radians).
    trunk_part_name : str
        Name of the trunk part in the cylinder JSON (e.g. "trunk_1", "trunk_3").
    num_poses : int
        Number of angular positions along the arc (default 6).
    num_z_levels : int
        Number of Z heights per angular position (default 6).
    central_angle : float
        Total arc angle in radians (default pi = semi-circle).
        The arc is centered on the reference camera.
    seed : int or None
        Random seed for reproducibility of Y offsets.

    Returns
    -------
    list of dict
        Each dict: {"cam_idx": int, "shot_idx": int,
                     "location": [x, y, z], "rotation_euler": [rx, ry, rz]}
        cam_idx is 1-based (angular position), shot_idx is 1-based (Z level).
    """
    if cylinder_json_path is None:
        cylinder_json_path = DEFAULT_CYLINDER_JSON
    if x_ref is None:
        x_ref = _X_REF
    if y_ref is None:
        y_ref = _Y_REF
    if z_ref is None:
        z_ref = _Z_REF
    if z_final is None:
        z_final = _Z_FINAL
    if rotation_ref is None:
        rotation_ref = list(_ROTATION_REF)

    if seed is not None:
        random.seed(seed)

    # 1. Trunk axis
    p_base, p_top = load_trunk_axis(cylinder_json_path, trunk_part_name)
    trunk_center_xy = p_base[:2]

    # 2. Radius = horizontal distance from reference camera to trunk center
    radius = math.sqrt(
        (x_ref - trunk_center_xy[0]) ** 2 +
        (y_ref - trunk_center_xy[1]) ** 2
    )

    # 3. Angle of the reference camera relative to trunk center
    theta_center = math.atan2(
        y_ref - trunk_center_xy[1],
        x_ref - trunk_center_xy[0],
    )

    # 4. Z values linearly spaced from z_ref to z_final
    if num_z_levels == 1:
        z_values = [z_ref]
    else:
        z_values = np.linspace(z_ref, z_final, num_z_levels).tolist()

    # 5. Theta values: arc centered on theta_center, spanning central_angle
    half_angle = central_angle / 2.0
    theta_start = theta_center - half_angle
    theta_end = theta_center + half_angle
    thetas = np.linspace(theta_start, theta_end, num_poses)

    ref_rx, ref_ry, ref_rz = rotation_ref

    poses = []
    for cam_i, theta in enumerate(thetas, start=1):
        cam_x = trunk_center_xy[0] + radius * math.cos(theta)
        cam_y = trunk_center_xy[1] + radius * math.sin(theta)

        # Small random Y offset — consistent across all Z levels for this cam
        y_offset = random.uniform(Y_OFFSET_MIN, Y_OFFSET_MAX)
        cam_y += y_offset

        # Camera rotation: ROTATION_REF with rz adjusted by angular offset
        # from the center of the arc.
        delta_theta = theta - theta_center
        rot = [ref_rx, ref_ry, ref_rz + delta_theta]

        for shot_i, z_val in enumerate(z_values, start=1):
            poses.append({
                "cam_idx": cam_i,
                "shot_idx": shot_i,
                "location": [float(cam_x), float(cam_y), float(z_val)],
                "rotation_euler": [float(rot[0]), float(rot[1]), float(rot[2])],
            })

    return poses


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    poses = generate_semicircle_poses()
    print(f"Generated {len(poses)} camera poses "
          f"({NUM_POSES} angles x {NUM_Z_LEVELS} Z-levels, "
          f"central_angle={math.degrees(CENTRAL_ANGLE):.0f}deg):\n")
    for p in poses:
        loc = p["location"]
        rot = p["rotation_euler"]
        print(
            f"  cam{p['cam_idx']} shot{p['shot_idx']:02d}: "
            f"loc=({loc[0]:+.4f}, {loc[1]:+.4f}, {loc[2]:+.4f})  "
            f"rot=({math.degrees(rot[0]):+.2f}deg, "
            f"{math.degrees(rot[1]):+.2f}deg, "
            f"{math.degrees(rot[2]):+.2f}deg)"
        )
