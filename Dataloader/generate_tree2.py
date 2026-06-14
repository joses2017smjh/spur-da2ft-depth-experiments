"""
generate_tree2.py — Same pipeline as generate_tree.py but:

- 6 camera Z-levels (X and Y fixed); Z linearly spaced from reference Z to Z_final.
- Rotation constant (reference).
- 3 bark texture variants applied per tree; output under rgb_4/<texture>/<tree_id>/ etc.
- Produces:
  - rgb_4/<texture>/<tree_id>: 6 center images per tree (one per Z level).
  - depth_4/<texture>/<tree_id>: 6 metric depth .npy files (center only).
  - ann_4/<texture>/<tree_id>: annotations for all 18 views (6 center + 12 optical flow).
  - mask_4/<texture>/<tree_id>: 18 tree-only masks (_c, _l, _r for each of 6 shots).
  - Optical_flow_4/<texture>/<tree_id>: 12 images per tree (_l and _r for each of 6 shots).

Bark texture sets: 4 variants (bark_brown, bark_brown_02, bark_willow, bark_willow_02). Each set lives in
TEXTURES_DIR/<folder>/ with <prefix>_diff_4k.jpg and <prefix>_nor_gl_4k.exr (normal map). .exr normals are loaded as Non-Color.
"""

import bpy
import os
import sys
import json
import math
import random
import re
import mathutils
import numpy as np

# Ensure Dataloader/ is on sys.path so `from move_camera import ...` works
# when Blender runs this script from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Base path: set COMPUTER_VISION_ROOT on HPC (e.g. /nfs/hpc/share/sanchej7/Computer_Vision)
_REPO_ROOT = os.environ.get("COMPUTER_VISION_ROOT", "/home/joses/Computer_Vision")
SCRIPT_DIR = os.path.join(_REPO_ROOT, "trees")
CONSTANT_CAMERA = os.path.join(_REPO_ROOT, "toy_example", "ann")
METADATA_DIR = os.path.join(_REPO_ROOT, "trees", "metadata")
PLY_DIR = os.path.join(_REPO_ROOT, "trees", "ply")
OUTPUT_DIR = os.environ.get("CV_OUTPUT_DIR", os.path.join(_REPO_ROOT, "Data", "full_spur"))

CAM = "Camera"
TREE_OBJ = "tree{}_TRUNK"
W, H = 1920, 1080

POST0_NAME = "post0"
POST1_NAME = "post1"
PASS_INDEX_TREE = 1
PASS_INDEX_BACKGROUND = 2
PASS_INDEX_BOX = 3          # camera-rect box — isolated so we can generate its mask

# Reference pose (first camera); X and Y stay fixed for all poses
X_REF = -9.820751190185547
Y_REF = -7.150839786529541
Z_REF = .85
Z_FINAL = 3.73232572555542
ROTATION_REF = [1.5395008325576782, 0.0, 3.194649314880371]

# Post axis tilt (degrees): rx 17.143, ry -0.0454, rz -0.0454 — applied to tree so trunk is parallel to posts
TREE_TILT_DEG = (-17.143, 0, 0)
TREE_TILT_RAD = (math.radians(TREE_TILT_DEG[0]), math.radians(TREE_TILT_DEG[1]), math.radians(TREE_TILT_DEG[2]))
# Background tree: same axis, rx negative (tilt other way)
TREE_TILT_BG_RAD = (-TREE_TILT_RAD[0], TREE_TILT_RAD[1], TREE_TILT_RAD[2])

# Background tree (one random other tree behind main): offset range in meters
# "Behind" = along camera view direction (from ROTATION_REF), not camera position. So it works the same
# whether camera is close (Y_REF -9.9) or far (Y_REF -5): bg = main_loc + back*forward + lateral*right.
TREE_BG_OBJ = "tree_bg_TRUNK"
BG_BACK_MIN, BG_BACK_MAX = .05, .1  # meters along view direction (past main tree)
BG_LATERAL_MIN, BG_LATERAL_MAX = 0.01, .04  # left/right of main tree (camera right vector)

# If set, only process trees whose tree_id contains this substring (e.g. "envy" for lpy_envy_*). None = all trees.
TREE_ID_FILTER = "envy"

# If set, replace the full PLY mesh with cylinder-built geometry.
# True = use metadata["hierarchy"]["root"][0] per tree (trunk); if RENDER_BRANCHES also True, include branch_* parts.
# A string (e.g. "trunk_1") = use that part name for all trees, and as fallback when root is missing.
# None = render full tree mesh.
RENDER_ONLY_PART = True
# When True and RENDER_ONLY_PART is True, include all hierarchy parts named "branch_1", "branch_2", ... "branch_xx".
# Env override CV_RENDER_BRANCHES (default 1) so the trunk-only generation script can disable branches without editing this file.
RENDER_BRANCHES = os.environ.get("CV_RENDER_BRANCHES", "1").lower() in ("1", "true", "yes")
# When True and RENDER_ONLY_PART is True, include all hierarchy parts named "spur_1", "spur_2", ... (full tree).
# Env override CV_RENDER_SPURS (default 1): set to 0 for trunk-only (full_trunk) renders; default 1 keeps full_spur.
RENDER_SPURS = os.environ.get("CV_RENDER_SPURS", "1").lower() in ("1", "true", "yes")

# Lateral offset for optical flow left/right (meters, in camera-right direction)
DX_OFFSET = 0.12

# Number of Z-level poses (6 center shots; at each level also render _l and _r for optical flow)
NUM_Z_LEVELS = 6

# Camera-space rectangle: when True, a dark-gray rectangle is placed at a fixed
# depth in front of the camera (CAMERA_RECT_DEPTH), perpendicular to the view
# direction, so it is always visible in the frame and moves with the camera.
ENABLE_CAMERA_RECT          = True
CAMERA_RECT_HEIGHT_M        = 0.07  # vertical extent of rect in camera space (meters)
CAMERA_RECT_WIDTH_M         = 0.07   # horizontal extent of rect in camera space (meters)
CAMERA_RECT_DEPTH           = .30      # meters in front of camera along view direction
# Vertical position in frame: -1.0 = bottom edge, 0.0 = center, 1.0 = top edge.
CAMERA_RECT_FRAME_V_OFFSET  = -.3
# Horizontal position in frame: -1.0 = left edge, 0.0 = center, 1.0 = right edge.
CAMERA_RECT_FRAME_H_OFFSET  = -.20
# Thickness of the 3-D box along the camera view direction (meters).
CAMERA_RECT_THICKNESS_M     = 0.15

# Semi-circular camera flag: when True, also render poses along an arc
# centered on the reference camera. Outputs go into cam1/ ... camN/ subdirs.
ENABLE_SEMICIRCLE_CAMERAS = True
NUM_SEMICIRCLE_POSES = 10
# Box-cam: same semicircle arc but WITH the camera rectangle in frame.
# Outputs go into box_cam1/ ... box_camN/ subdirs.
NUM_BOX_CAM_POSES = int(os.environ.get("CV_NUM_BOX_CAM_POSES", "8"))
ENABLE_OPTICAL_FLOW = True
ENABLE_BACKGROUND_TREE = False
ENABLE_DARK_LIGHT_CAM = False
FORCE_RENDER = os.environ.get("CV_FORCE_RENDER", "0").lower() in ("1", "true", "yes")
# Skip render_tree_2 and the non-box semicircle pass — useful for box-cam
# smoke tests where we only need the _l/_r box mask renders.
BOXCAM_ONLY = os.environ.get("CV_BOXCAM_ONLY", "0").lower() in ("1", "true", "yes")


# Total arc angle (radians).  pi = semi-circle (90 deg left + 90 deg right).
# e.g. math.radians(30) for 15 deg left + 15 deg right.
import math as _math
SEMICIRCLE_CENTRAL_ANGLE = _math.radians(100)

# Lighting mode: when True a low-angle warm sun is added and the world sky is
# darkened to simulate late-afternoon / dusk conditions.
# When False the original solid overcast sky (no sun lamp) is used.
LOW_LIGHT_MODE = False

# Output base dirs (_4): under each we have <texture_name>/<tree_id>/
RGB_DIR_4 = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_DIR_4 = os.path.join(OUTPUT_DIR, "depth")
ANN_DIR_4 = os.path.join(OUTPUT_DIR, "ann")
MASK_DIR_4 = os.path.join(OUTPUT_DIR, "mask")
BOX_MASK_DIR = os.path.join(OUTPUT_DIR, "box_mask")   # camera-rect box masks
OPTICAL_FLOW_DIR_4 = os.path.join(OUTPUT_DIR, "Optical_flow")
CYLINDERS_WORLD_DIR = os.path.join(OUTPUT_DIR, "cylinders_world")

# Base path for texture sets (fixed path so it never doubles "Simple" when OUTPUT_DIR is Data/Simple)
TEXTURES_DIR = os.path.join(_REPO_ROOT, "Data",  "textures")
# Optional override: os.path.join(OUTPUT_DIR, "textures") or "/path/to/your/textures"

# Bark texture sets: name (output subdir), folder under TEXTURES_DIR, file prefix inside folder
# find_texture_paths looks for <prefix>_diff_4k.jpg and <prefix>_nor_gl_4k.exr in TEXTURES_DIR/<folder>/
BARK_TEXTURES = [    
    {"name": "bark_brown", "folder": "bark_brown", "prefix": "bark_brown"},
    {"name": "bark_brown_02", "folder": "bark_brown_02", "prefix": "bark_brown_02"},
    {"name": "bark_willow", "folder": "bark_willow", "prefix": "bark_willow"},
    {"name": "bark_willow_02", "folder": "bark_willow_02", "prefix": "bark_willow_02"},
]

# World-space view direction and right from camera euler (Blender: cam looks -Z local)
def _euler_xyz_to_R(e):
    rx, ry, rz = e[0], e[1], e[2]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ]

def transform_points_world(tree_obj, points):
    """
    Convert Nx3 cylinder points from tree local space → Blender world space.
    """
    M = tree_obj.matrix_world
    out = []
    for x, y, z in points:
        v = mathutils.Vector((x, y, z, 1.0))
        vw = M @ v
        out.append([vw.x, vw.y, vw.z])
    return out


def transform_direction_world(tree_obj, direction):
    """
    Transform a unit direction vector from tree local → world (rotation only, then re-normalize).
    """
    M = tree_obj.matrix_world
    R = M.to_3x3()
    d = mathutils.Vector(direction)
    dw = R @ d
    dw.normalize()
    return [dw.x, dw.y, dw.z]


def get_cylinder_data_from_metadata(meta):
    """
    Return list of cylinder records in tree local frame.
    Each record: {"centroid": [x,y,z], "orientation": [x,y,z], "radius": r, "length": l, "part_name": str}.
    Supports meta["cylinder_data"] (full records) or meta["cylinders"] (list of centroids only; orientation null).
    """
    if "cylinder_data" in meta:
        out = []
        for data in meta["cylinder_data"].values():
            rec = {
                "centroid": list(data["centroid"]),
                "orientation": list(data.get("orientation", [0, 0, 1])),
                "radius": data.get("radius"),
                "length": data.get("length"),
                "part_name": data.get("part_name", ""),
            }
            out.append(rec)
        return out
    if "cylinders" in meta:
        return [
            {"centroid": list(pt), "orientation": [0, 0, 1], "radius": None, "length": None, "part_name": ""}
            for pt in meta["cylinders"]
        ]
    return []


def transform_cylinders_world(tree_obj, cylinders_local):
    """
    Transform list of cylinder records (centroid + orientation) from tree local → world.
    Returns list of dicts: centroid (world), orientation (world, unit), radius, length, part_name.
    """
    out = []
    for c in cylinders_local:
        cent_world = transform_points_world(tree_obj, [c["centroid"]])[0]
        ori_world = transform_direction_world(tree_obj, c["orientation"])
        out.append({
            "centroid": cent_world,
            "orientation": ori_world,
            "radius": c.get("radius"),
            "length": c.get("length"),
            "part_name": c.get("part_name", ""),
        })
    return out


def _cylinder_verts_faces(radius, length, segments=16):
    """Return (verts, faces) for a cylinder in local space: axis Z, centered at origin, Z from -length/2 to length/2."""
    verts = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        c, s = math.cos(angle), math.sin(angle)
        verts.append((radius * c, radius * s, -length / 2.0))
        verts.append((radius * c, radius * s, length / 2.0))
    faces = []
    # bottom cap
    faces.append(tuple(range(0, 2 * segments, 2)))
    # top cap
    faces.append(tuple(range(1, 2 * segments + 1, 2)))
    # side quads
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    return verts, faces


def mesh_from_cylinders_local(cylinders_local, name="Trunk1Mesh"):
    """
    Build a single Blender mesh from a list of cylinder records in tree local space.
    Each record: centroid [x,y,z], orientation [x,y,z] (unit axis), radius, length.
    Skips cylinders with missing radius/length. Returns bpy.types.Mesh.
    """
    z_axis = mathutils.Vector((0, 0, 1))
    all_verts = []
    all_faces = []
    for c in cylinders_local:
        r, L = c.get("radius"), c.get("length")
        if r is None or L is None or r <= 0 or L <= 0:
            continue
        centroid = mathutils.Vector(c["centroid"])
        orientation = mathutils.Vector(c["orientation"])
        orientation.normalize()
        verts, faces = _cylinder_verts_faces(float(r), float(L))
        # Rotation from Z to orientation
        if orientation.length_squared > 1e-10:
            quat = z_axis.rotation_difference(orientation)
            rot_mat = quat.to_matrix()
        else:
            rot_mat = mathutils.Matrix.Identity(3)
        for v in verts:
            p = mathutils.Vector(v)
            p = rot_mat @ p
            p = p + centroid
            all_verts.append(p)
        base = len(all_verts) - len(verts)
        for f in faces:
            all_faces.append(tuple(base + i for i in f))
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(all_verts, [], all_faces)
    mesh.update()
    return mesh


def camera_forward_right_from_euler(rot_euler_xyz):
    """Return (forward, right) world vectors; forward = view direction, right = camera right."""
    R = _euler_xyz_to_R(rot_euler_xyz)
    # Blender camera looks along -Z in local
    forward = (-R[0][2], -R[1][2], -R[2][2])
    right = (R[0][0], R[1][0], R[2][0])
    return forward, right


# Intrinsics (same as reference; 1920x1080)
K_REF = [
    [2666.666666666667, 0.0, 960.0],
    [0.0, 1500.0, 540.0],
    [0.0, 0.0, 1.0],
]


def get_root_name(meta_pth):
    try:
        with open(meta_pth, "r") as f:
            metadata = json.load(f)
        root = metadata.get("hierarchy", {}).get("root", [])
        if root:
            return root[0]
    except Exception as e:
        print(f"error reading metadata {meta_pth}: {e}")
    return None


def find_all_metadata_files():
    if not os.path.exists(METADATA_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(METADATA_DIR)):
        if fname.endswith("_metadata.json"):
            tree_id = fname.replace("_metadata.json", "")
            metadata_path = os.path.join(METADATA_DIR, fname)
            match = re.search(r"(\d+)$", tree_id)
            file_num = int(match.group(1)) if match else 0
            out.append((tree_id, metadata_path, file_num))
    return out


def _blender_path(path):
    """Return absolute path with forward slashes so Blender loads reliably (3.6)."""
    return os.path.normpath(os.path.abspath(path)).replace("\\", "/")


def find_texture_paths(texture_entry):
    """Return (diff_path, normal_path) for a texture set. normal_path may be None if .exr not found.
    texture_entry: dict with 'folder' and 'prefix'. Looks for files containing the prefix
    with _diff_4k and _nor_gl_4k patterns. Falls back to glob matching if exact name not found."""
    folder = texture_entry["folder"]
    prefix = texture_entry["prefix"]
    base = os.path.join(TEXTURES_DIR, folder)
    diff_path = None
    # Try exact match first: <prefix>_diff_4k.ext
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(base, f"{prefix}_diff_4k{ext}")
        if os.path.isfile(p):
            diff_path = p
            break
    # Fallback: find any file starting with prefix and containing _diff_4k
    if diff_path is None and os.path.isdir(base):
        for fname in sorted(os.listdir(base)):
            if fname.startswith(prefix) and "_diff_4k" in fname:
                diff_path = os.path.join(base, fname)
                break
    normal_path = os.path.join(base, f"{prefix}_nor_gl_4k.exr")
    if not os.path.isfile(normal_path):
        # Fallback: find any file starting with prefix and containing _nor_gl_4k
        normal_path = None
        if os.path.isdir(base):
            for fname in sorted(os.listdir(base)):
                if fname.startswith(prefix) and "_nor_gl_4k" in fname:
                    normal_path = os.path.join(base, fname)
                    break
    return diff_path, normal_path


def make_material_from_textures(diff_path, normal_path=None, mat_name=None):
    """Create a Blender material: Principled BSDF with diffuse image and optional normal map (.exr).
    Uses forward-slash paths and sets normal map to Non-Color for Blender 3.6."""
    if mat_name is None:
        mat_name = "Bark_" + os.path.splitext(os.path.basename(diff_path))[0]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (200, 0)

    # Use object-space mapping + box projection to avoid UV seams/patchwork when meshes have no good UVs.
    # This makes the bark mapping consistent across disconnected trunk pieces ("cylinders").
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    mapping.vector_type = "POINT"
    # Tweak scale to taste; lower = larger features. Keep uniform for bark.
    mapping.inputs["Scale"].default_value = (0.35, 0.35, 0.35)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    tex_diff = nodes.new("ShaderNodeTexImage")
    tex_diff.location = (-400, 0)
    tex_diff.projection = "BOX"
    tex_diff.projection_blend = 0.2
    diff_abs = _blender_path(diff_path)
    try:
        img_diff = bpy.data.images.load(diff_abs, check_existing=True)
        tex_diff.image = img_diff
        # Diffuse: keep default sRGB
        if img_diff.size[0] > 0 and img_diff.size[1] > 0:
            print(f"    Loaded diffuse: {os.path.basename(diff_path)} ({img_diff.size[0]}x{img_diff.size[1]})")
        else:
            print(f"    Warning: diffuse image has zero size: {diff_path}")
    except Exception as e:
        print(f"Failed to load diffuse texture {diff_abs}: {e}")
        bpy.data.materials.remove(mat)
        return None
    links.new(mapping.outputs["Vector"], tex_diff.inputs["Vector"])
    links.new(tex_diff.outputs["Color"], principled.inputs["Base Color"])

    if normal_path and os.path.isfile(normal_path):
        tex_nor = nodes.new("ShaderNodeTexImage")
        tex_nor.location = (-400, -220)
        tex_nor.projection = "BOX"
        tex_nor.projection_blend = 0.2
        nor_abs = _blender_path(normal_path)
        try:
            img_nor = bpy.data.images.load(nor_abs, check_existing=True)
            tex_nor.image = img_nor
            # Normal maps must be Non-Color (no sRGB) in Blender 3.6
            img_nor.colorspace_settings.name = "Non-Color"
            if hasattr(img_nor.colorspace_settings, "is_data"):
                img_nor.colorspace_settings.is_data = True
            if img_nor.size[0] > 0 and img_nor.size[1] > 0:
                print(f"    Loaded normal:  {os.path.basename(normal_path)} ({img_nor.size[0]}x{img_nor.size[1]})")
            else:
                print(f"    Warning: normal image has zero size: {normal_path}")
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.location = (-180, -220)
            links.new(mapping.outputs["Vector"], tex_nor.inputs["Vector"])
            links.new(tex_nor.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        except Exception as e:
            print(f"Warning: could not load normal map {nor_abs}: {e}")

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    return mat


def load_ply_into_blender(ply_pth, object_name):
    if not os.path.exists(ply_pth):
        print(f"PLY does not exist: {ply_pth}")
        return None
    bpy.ops.object.select_all(action="DESELECT")
    # Blender 4.2+ moved PLY import to wm.ply_import; fall back to legacy operator
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=ply_pth)
    else:
        bpy.ops.import_mesh.ply(filepath=ply_pth)
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        print(f"Import failed for {ply_pth}")
        return None
    obj.name = object_name
    obj.hide_viewport = False
    obj.hide_render = False
    print(f"Loaded {os.path.basename(ply_pth)} as {object_name}")
    return obj


def ensure_uv_layer(mesh_obj):
    """If mesh has no UV layer, add one using Smart UV Project so image textures can map correctly."""
    if not getattr(mesh_obj.data, "uv_layers", None) or len(mesh_obj.data.uv_layers) == 0:
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh_obj.select_set(False)
        print(f"  Added UV layer (Smart UV Project) to {mesh_obj.name} for texture mapping.")


def fix_world_background():
    """Replace world background with solid color. Blend file may have HDRI/image with broken path on HPC → whole frame purple."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    if LOW_LIGHT_MODE:
        bg.inputs["Color"].default_value = (0.10, 0.13, 0.20, 1.0)  # Dark evening sky
        bg.inputs["Strength"].default_value = 0.144  # 0.12 * 1.2 (+20%)
    else:
        bg.inputs["Color"].default_value = (0.7, 0.75, 0.82, 1.0)  # Soft overcast sky
        bg.inputs["Strength"].default_value = 0.5
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (200, 0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    mode_label = "low-light evening" if LOW_LIGHT_MODE else "overcast"
    print(f"  World background set to solid color ({mode_label})", flush=True)


def setup_low_light_sun():
    """Add a single low-angle warm sun lamp to simulate late-afternoon / dusk.

    Only called when LOW_LIGHT_MODE is True.

    Settings:
      - Elevation ~12 degrees above horizon (nearly grazing, long shadows).
      - Warm golden-orange colour (1.0, 0.68, 0.35).
      - Energy 1.2 — noticeably dimmer than a midday sun (typically 4-7 in Cycles).
      - Sun disk angle 8 deg — slightly soft shadow edges.
    Any existing sun lamps in the scene are removed first so the blend file's
    default lighting does not interfere.
    """
    # Remove any pre-existing sun lamps from the blend file
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.name = "LowLightSun"

    # Low elevation: rotate X ≈ 78 deg so sun is only ~12 deg above horizon.
    # Rotate Z ≈ 45 deg so it comes from a diagonal (side-front) direction.
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = (math.radians(78), 0.0, math.radians(45))

    sun.data.energy = 2.903  # prev 2.0736 * 1.4 (+40%)
    sun.data.color = (1.0, 0.68, 0.35)          # warm golden-orange
    sun.data.angle = math.radians(8)             # slightly soft shadows

    print(
        f"  Low-light sun added: energy={sun.data.energy}, "
        f"elevation≈12°, color=(1.0, 0.68, 0.35)",
        flush=True,
    )


def fix_world_background_dark():
    """Switch world background to dark evening sky (used for dark_light_cam pass)."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.10, 0.13, 0.20, 1.0)  # Dark evening sky
    bg.inputs["Strength"].default_value = 0.2903  # prev 0.20736 * 1.4 (+40%)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (200, 0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    print(f"  World background set to dark evening sky (+40% brightness)", flush=True)


def remove_low_light_sun():
    """Remove the low-light sun lamp (restore to normal overcast lighting)."""
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.data.type == "SUN" and obj.name == "LowLightSun":
            bpy.data.objects.remove(obj, do_unlink=True)
    print("  Low-light sun removed", flush=True)


def make_ground_material_with_bumps(diff_path, normal_path, disp_path=None):
    """Create ground material with diffuse, normal (stronger), and optional displacement for bumps."""
    mat = bpy.data.materials.new(name="Ground_dirt")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (200, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    mapping.vector_type = "POINT"
    mapping.inputs["Scale"].default_value = (0.15, 0.15, 0.15)  # Ground scale for dirt tiling
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Diffuse
    tex_diff = nodes.new("ShaderNodeTexImage")
    tex_diff.location = (-400, 100)
    tex_diff.projection = "BOX"
    tex_diff.projection_blend = 0.2
    diff_abs = _blender_path(diff_path)
    try:
        img_diff = bpy.data.images.load(diff_abs, check_existing=True)
        tex_diff.image = img_diff
    except Exception as e:
        print(f"  Failed to load ground diffuse {diff_abs}: {e}", flush=True)
        bpy.data.materials.remove(mat)
        return None
    links.new(mapping.outputs["Vector"], tex_diff.inputs["Vector"])
    links.new(tex_diff.outputs["Color"], principled.inputs["Base Color"])

    # Normal map + Bump (from height/displacement texture) for visible bumps.
    # Bump works without subdivided geometry; displacement would need a subdivided mesh.
    normal_input = None
    if normal_path and os.path.isfile(normal_path):
        tex_nor = nodes.new("ShaderNodeTexImage")
        tex_nor.location = (-400, -120)
        tex_nor.projection = "BOX"
        tex_nor.projection_blend = 0.2
        nor_abs = _blender_path(normal_path)
        try:
            img_nor = bpy.data.images.load(nor_abs, check_existing=True)
            tex_nor.image = img_nor
            img_nor.colorspace_settings.name = "Non-Color"
            if hasattr(img_nor.colorspace_settings, "is_data"):
                img_nor.colorspace_settings.is_data = True
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.location = (-180, -120)
            normal_map.inputs["Strength"].default_value = 2.2  # Stronger for visible bumps
            links.new(mapping.outputs["Vector"], tex_nor.inputs["Vector"])
            links.new(tex_nor.outputs["Color"], normal_map.inputs["Color"])
            normal_input = normal_map.outputs["Normal"]
        except Exception as e:
            print(f"  Warning: could not load ground normal {nor_abs}: {e}", flush=True)

    # Bump from displacement/height map — works on any mesh, no subdivision needed
    if disp_path and os.path.isfile(disp_path):
        tex_disp = nodes.new("ShaderNodeTexImage")
        tex_disp.location = (-400, -340)
        tex_disp.projection = "BOX"
        tex_disp.projection_blend = 0.2
        disp_abs = _blender_path(disp_path)
        try:
            img_disp = bpy.data.images.load(disp_abs, check_existing=True)
            tex_disp.image = img_disp
            img_disp.colorspace_settings.name = "Non-Color"
            if hasattr(img_disp.colorspace_settings, "is_data"):
                img_disp.colorspace_settings.is_data = True
            bump_node = nodes.new("ShaderNodeBump")
            bump_node.location = (-180, -340)
            bump_node.inputs["Strength"].default_value = 0.15  # Visible bump height
            bump_node.inputs["Distance"].default_value = 0.1
            links.new(mapping.outputs["Vector"], tex_disp.inputs["Vector"])
            links.new(tex_disp.outputs["Color"], bump_node.inputs["Height"])
            if normal_input is not None:
                links.new(normal_input, bump_node.inputs["Normal"])
            links.new(bump_node.outputs["Normal"], principled.inputs["Normal"])
        except Exception as e:
            print(f"  Warning: could not load ground displacement {disp_abs}: {e}", flush=True)
            if normal_input is not None:
                links.new(normal_input, principled.inputs["Normal"])
    elif normal_input is not None:
        links.new(normal_input, principled.inputs["Normal"])

    return mat


def fix_ground_material():
    """Replace ground object material with dirt_floor textures from TEXTURES_DIR. Blend stores paths from local machine → purple on HPC."""
    base = os.path.join(TEXTURES_DIR, "dirt_floor")
    diff_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(base, "dirt_floor_diff_4k" + ext)
        if os.path.isfile(p):
            diff_path = p
            break
    if not diff_path:
        print("  Warning: dirt_floor texture not found; ground may show purple", flush=True)
        return
    normal_path = os.path.join(base, "dirt_floor_nor_gl_4k.exr")
    disp_path = os.path.join(base, "dirt_floor_disp_4k.png")
    mat = make_ground_material_with_bumps(diff_path, normal_path, disp_path)
    if mat is None:
        return
    for obj in bpy.data.objects:
        if "ground" in obj.name.lower() and obj.type == "MESH":
            if not obj.data.materials:
                obj.data.materials.append(mat)
            else:
                obj.data.materials.clear()
                obj.data.materials.append(mat)
            for poly in obj.data.polygons:
                poly.material_index = 0
            print(f"  Ground material replaced for {obj.name} (with bumps)", flush=True)


def remove_placeholder_objects():
    """Remove blend template placeholders (Cylinder.*, Cube) that cause purple tint when textures are missing on HPC."""
    names_to_remove = [
        obj.name for obj in list(bpy.data.objects)
        if obj.name == "Cube" or obj.name.startswith("Cylinder")
    ]
    if names_to_remove:
        print(f"  Removing placeholders: {names_to_remove}", flush=True)
    for name in names_to_remove:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
            print(f"  Removed: {name}", flush=True)
    if names_to_remove:
        bpy.context.view_layer.update()


def remove_all_tree_objects():
    to_remove = []
    for obj in bpy.data.objects:
        if obj.name.startswith("tree") and ("_TRUNK" in obj.name or "_BRANCH" in obj.name or "_SPUR" in obj.name):
            to_remove.append(obj)
        if "_TRUNK" in obj.name and "lpy_envy" in obj.name.lower():
            to_remove.append(obj)
        if "_TRUNK" in obj.name and "lpy_ufo" in obj.name.lower():
            to_remove.append(obj)
    seen = set()
    for obj in to_remove:
        if obj.name not in seen:
            seen.add(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)


def helper_all_tree_objects(tree_idx, remove_envy, remove_ufo):
    root = f"tree{tree_idx}"
    to_delete = [
        obj for obj in bpy.data.objects
        if obj.name.startswith(root) and any(t in obj.name for t in ("_TRUNK", "_BRANCH", "_SPUR"))
    ]
    if remove_envy:
        to_delete += [obj for obj in bpy.data.objects if "lpy_envy" in obj.name.lower() and "_TRUNK" in obj.name]
    if remove_ufo:
        to_delete += [obj for obj in bpy.data.objects if "lpy_ufo" in obj.name.lower() and "_TRUNK" in obj.name]
    seen = set()
    for obj in to_delete:
        if obj.name not in seen:
            seen.add(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)


def get_tree_object_for_metadata(tree_id, metadata_path, tree_idx, saved_tree_data=None, texture_entry=None):
    expected_name = TREE_OBJ.format(tree_idx)
    ply_pth = os.path.join(PLY_DIR, f"{tree_id}.ply")
    original_location = None
    original_material = None
    if saved_tree_data and tree_idx in saved_tree_data:
        original_location, original_material = saved_tree_data[tree_idx]
    else:
        existing = bpy.data.objects.get(expected_name)
        if existing:
            original_location = existing.matrix_world.to_translation().copy()
            if getattr(existing.data, "materials", None) and existing.data.materials:
                original_material = existing.data.materials[0]

    remove_all_tree_objects()

    # If we only render cylinders, DO NOT import PLY (Blender 4.2 has no PLY importer)
    if RENDER_ONLY_PART:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        cylinders_local = get_cylinder_data_from_metadata(meta)
        if not cylinders_local:
            print(f"  ERROR: no cylinder_data/cylinders in metadata for {tree_id}; cannot build mesh. Skipping.")
            return None

        # Walk the hierarchy from root to collect only connected parts.
        # This avoids floating geometry from orphan spurs/branches.
        hierarchy = meta.get("hierarchy", {})
        trunk_name = hierarchy.get("root", [None])[0]

        # Collect part names by walking the tree
        parts_to_render = set()
        if trunk_name:
            parts_to_render.add(trunk_name)
        else:
            # Fallback: grab anything starting with "trunk"
            for c in cylinders_local:
                pn = (c.get("part_name") or "").lower()
                if pn.startswith("trunk"):
                    parts_to_render.add(c.get("part_name"))

        if RENDER_BRANCHES or RENDER_SPURS:
            # BFS: only add children whose type we want, only recurse into structural parts
            queue = list(parts_to_render)
            while queue:
                parent = queue.pop(0)
                for child in hierarchy.get(parent, []):
                    cl = child.lower()
                    if cl.startswith("branch_") or cl.startswith("nontrunk_"):
                        if RENDER_BRANCHES:
                            parts_to_render.add(child)
                            queue.append(child)
                    elif cl.startswith("spur_"):
                        if RENDER_SPURS:
                            parts_to_render.add(child)
                    elif cl.startswith("trunk"):
                        parts_to_render.add(child)
                        queue.append(child)

        cylinders_part = [
            c for c in cylinders_local
            if c.get("part_name") in parts_to_render
        ]

        mesh = mesh_from_cylinders_local(cylinders_part, name=f"Trunk_{tree_id}")
        tree_obj = bpy.data.objects.new(expected_name, mesh)
        bpy.context.collection.objects.link(tree_obj)
        kept_parts = set((c.get("part_name") or "") for c in cylinders_part)
        all_parts = set((c.get("part_name") or "") for c in cylinders_local)
        print(f"  Built mesh from {len(cylinders_part)}/{len(cylinders_local)} cylinders. Kept parts: {sorted(kept_parts)}. Filtered out: {sorted(all_parts - kept_parts)}")
    else:
        # Full-tree rendering requires PLY import (won't work on Blender 4.2 if PLY addon missing)
        if not os.path.exists(ply_pth):
            print(f"PLY not found: {ply_pth}")
            return None
        tree_obj = load_ply_into_blender(ply_pth, expected_name)
        if tree_obj is None:
            return None

    if original_location is not None:
        tree_obj.location = original_location
    elif saved_tree_data and 0 in saved_tree_data:
        tree_obj.location = saved_tree_data[0][0]
    else:
        tree_obj.location = (0.0, 0.0, 0.0)
    # Tilt tree to match post axis (parallel to posts)
    tree_obj.rotation_mode = "XYZ"
    tree_obj.rotation_euler = TREE_TILT_RAD
    if texture_entry:
        diff_path, normal_path = find_texture_paths(texture_entry)
        print(f"  Texture set '{texture_entry['name']}': diff={diff_path}, normal={normal_path}")
        if diff_path:
            # PLY often has no UVs; add a UV layer so image textures can display
            ensure_uv_layer(tree_obj)
            mat = make_material_from_textures(
                diff_path, normal_path=normal_path, mat_name=f"Bark_{texture_entry['name']}"
            )
            if mat:
                tree_obj.data.materials.clear()
                tree_obj.data.materials.append(mat)
                # Force every face to use slot 0 so no face keeps an old material index
                for poly in tree_obj.data.polygons:
                    poly.material_index = 0
                print(f"  Assigned material '{mat.name}' to {tree_obj.name} (render object). Materials now: {[m.name for m in tree_obj.data.materials]}")
            else:
                print(f"  ERROR: make_material_from_textures failed; tree will keep PLY default (not applying original_material).")
        else:
            print(f"  Warning: texture set '{texture_entry['name']}' not found in {TEXTURES_DIR}, using existing material")
            if original_material:
                tree_obj.data.materials.clear()
                tree_obj.data.materials.append(original_material)
        # When texture_entry is set we never re-apply original_material after the block above.
    elif original_material:
        tree_obj.data.materials.clear()
        tree_obj.data.materials.append(original_material)
    if not texture_entry and saved_tree_data:
        for fall_idx in (0, 1):
            if fall_idx in saved_tree_data and saved_tree_data[fall_idx][1] and not original_material:
                tree_obj.data.materials.clear()
                tree_obj.data.materials.append(saved_tree_data[fall_idx][1])
                break
    tree_obj.hide_viewport = False
    tree_obj.hide_render = False
    bpy.context.view_layer.objects.active = tree_obj
    tree_obj.select_set(True)
    bpy.context.view_layer.update()
    return tree_obj


def load_background_tree(tree_id_bg, main_tree_obj, texture_entry):
    """Load one other tree behind and left/right of main tree; oriented to post axis with rx negative. Returns object or None."""
    ply_pth = os.path.join(PLY_DIR, f"{tree_id_bg}.ply")
    if not os.path.exists(ply_pth):
        return None
    existing_bg = bpy.data.objects.get(TREE_BG_OBJ)
    if existing_bg:
        bpy.data.objects.remove(existing_bg, do_unlink=True)
    bg_obj = load_ply_into_blender(ply_pth, TREE_BG_OBJ)
    if bg_obj is None:
        return None
    main_loc = main_tree_obj.matrix_world.to_translation()
    forward, right = camera_forward_right_from_euler(ROTATION_REF)
    back = random.uniform(BG_BACK_MIN, BG_BACK_MAX)
    lateral = random.uniform(BG_LATERAL_MIN, BG_LATERAL_MAX)
    bg_obj.location = (
        main_loc.x + back * forward[0] + lateral * right[0],
        main_loc.y + back * forward[1] + lateral * right[1],
        main_loc.z + back * forward[2] + lateral * right[2],
    )
    bg_obj.rotation_mode = "XYZ"
    bg_obj.rotation_euler = TREE_TILT_BG_RAD
    if texture_entry:
        diff_path, normal_path = find_texture_paths(texture_entry)
        if diff_path:
            ensure_uv_layer(bg_obj)
            mat = make_material_from_textures(
                diff_path, normal_path=normal_path, mat_name=f"Bark_bg_{texture_entry['name']}"
            )
            if mat:
                bg_obj.data.materials.clear()
                bg_obj.data.materials.append(mat)
                for poly in bg_obj.data.polygons:
                    poly.material_index = 0
    bg_obj.pass_index = PASS_INDEX_BACKGROUND
    bg_obj.hide_viewport = False
    bg_obj.hide_render = False
    bpy.context.view_layer.update()
    return bg_obj


def get_z_levels():
    """Return list of Z values linearly spaced from Z_REF to Z_FINAL (inclusive), length NUM_Z_LEVELS."""
    if NUM_Z_LEVELS == 1:
        return [Z_REF]
    return [Z_REF + (Z_FINAL - Z_REF) * i / (NUM_Z_LEVELS - 1) for i in range(NUM_Z_LEVELS)]


def get_background_objects():
    names = []
    for i in range(10):
        if bpy.data.objects.get(f"post{i}"):
            names.append(f"post{i}")
    for obj in bpy.data.objects:
        if "wire" in obj.name.lower() and obj.type in {"MESH", "CURVE"}:
            names.append(obj.name)
    for obj in bpy.data.objects:
        if "ground" in obj.name.lower() and obj.type == "MESH":
            names.append(obj.name)
    return names


def set_pass_indices(tree_obj, union_objects, background_tree_obj=None):
    if tree_obj:
        tree_obj.pass_index = PASS_INDEX_TREE
    if background_tree_obj:
        background_tree_obj.pass_index = PASS_INDEX_BACKGROUND
    for name in union_objects:
        obj = bpy.data.objects.get(name)
        if obj:
            obj.pass_index = PASS_INDEX_BACKGROUND


def setup_mask_tree_only(scene, mask_output_dir, mask_prefix):
    """Compositor: only tree mask (ID==1), no union. For mask_4.

    On first call the full node tree is built. On subsequent calls with the
    same topology (tree-only, no box) only the output paths are updated,
    skipping the expensive nodes.clear() + full rebuild.
    """
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    scene.use_nodes = True
    nt = scene.node_tree

    # Fast path: tree is already set up for tree-only mode — just update paths.
    out_tree = nt.nodes.get("MaskTreeOut")
    if (out_tree is not None
            and nt.nodes.get("DepthFileOutput") is None  # no leftover box/depth nodes
            and nt.nodes.get("MaskBoxOut") is None):
        out_tree.base_path = mask_output_dir
        out_tree.file_slots[0].path = f"{mask_prefix}_tree_"
        return

    # Full build (first call or after a topology change).
    nt.nodes.clear()
    nodes = nt.nodes
    links = nt.links
    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    idx_sock = rl.outputs.get("IndexOB")
    if idx_sock is None:
        raise RuntimeError("RenderLayers has no IndexOB output.")
    tree_mask = nodes.new("CompositorNodeIDMask")
    tree_mask.index = PASS_INDEX_TREE
    tree_mask.location = (-150, 100)
    links.new(idx_sock, tree_mask.inputs["ID value"])
    out_tree = nodes.new("CompositorNodeOutputFile")
    out_tree.name = "MaskTreeOut"
    out_tree.base_path = mask_output_dir
    out_tree.format.file_format = "PNG"
    out_tree.format.color_mode = "BW"
    out_tree.file_slots[0].path = f"{mask_prefix}_tree_"
    out_tree.location = (200, 80)
    links.new(tree_mask.outputs["Alpha"], out_tree.inputs[0])


def setup_mask_tree_and_box(scene, mask_output_dir, mask_prefix,
                            box_mask_output_dir, box_mask_prefix):
    """Compositor: tree mask (ID==1) + box mask (ID==3) in one pass."""
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    nodes = nt.nodes
    links = nt.links
    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    idx_sock = rl.outputs.get("IndexOB")
    if idx_sock is None:
        raise RuntimeError("RenderLayers has no IndexOB output.")
    # Tree mask
    tree_mask = nodes.new("CompositorNodeIDMask")
    tree_mask.index = PASS_INDEX_TREE
    tree_mask.location = (-150, 120)
    links.new(idx_sock, tree_mask.inputs["ID value"])
    out_tree = nodes.new("CompositorNodeOutputFile")
    out_tree.base_path = mask_output_dir
    out_tree.format.file_format = "PNG"
    out_tree.format.color_mode = "BW"
    out_tree.file_slots[0].path = f"{mask_prefix}_tree_"
    out_tree.location = (200, 120)
    links.new(tree_mask.outputs["Alpha"], out_tree.inputs[0])
    # Box mask
    box_mask = nodes.new("CompositorNodeIDMask")
    box_mask.index = PASS_INDEX_BOX
    box_mask.location = (-150, -80)
    links.new(idx_sock, box_mask.inputs["ID value"])
    out_box = nodes.new("CompositorNodeOutputFile")
    out_box.base_path = box_mask_output_dir
    out_box.format.file_format = "PNG"
    out_box.format.color_mode = "BW"
    out_box.file_slots[0].path = f"{box_mask_prefix}_box_"
    out_box.location = (200, -80)
    links.new(box_mask.outputs["Alpha"], out_box.inputs[0])


import glob as _glob

def rename_mask_output(mask_dir, prefix, suffix="_tree_"):
    """Rename Blender compositor output files to remove the frame-number suffix.

    Blender's CompositorNodeOutputFile always appends a frame number, producing
    files like  <prefix>_tree_0001.png.  This renames that to <prefix>.png.
    """
    pattern = os.path.join(mask_dir, f"{prefix}{suffix}*.png")
    for src in _glob.glob(pattern):
        dst = os.path.join(mask_dir, f"{prefix}.png")
        try:
            os.rename(src, dst)
        except OSError as e:
            print(f"  Warning: could not rename {src} -> {dst}: {e}")


def ensure_tree_dirs_4(tree_id, texture_name):
    """Create rgb_4/<texture>/<tree_id>, depth_4/<texture>/<tree_id>, etc."""
    rgb_dir = os.path.join(RGB_DIR_4, texture_name, tree_id)
    depth_dir = os.path.join(DEPTH_DIR_4, texture_name, tree_id)
    ann_dir = os.path.join(ANN_DIR_4, texture_name, tree_id)
    mask_dir = os.path.join(MASK_DIR_4, texture_name, tree_id)
    flow_dir = os.path.join(OPTICAL_FLOW_DIR_4, texture_name, tree_id)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    return rgb_dir, depth_dir, ann_dir, mask_dir, flow_dir


def set_cycles_gpu(scene, prefer_optix=True):
    """Enable Cycles GPU rendering (OptiX for RTX, else CUDA). Falls back to CPU if no GPU available.
    If you see 'CUDA cuInit: Unknown error' then GPU is disabled by Blender/driver; try opening Blender
    GUI once, go to Edit > Preferences > System > Cycles, enable your GPU, and save preferences."""
    prefs = bpy.context.preferences
    addon = prefs.addons.get("cycles")
    if not addon:
        print("Cycles addon not found; using CPU.")
        return
    cycles_prefs = addon.preferences
    device_types = ["OPTIX", "CUDA"] if prefer_optix else ["CUDA", "OPTIX"]
    for device_type in device_types:
        try:
            scene.cycles.device = "GPU"
            cycles_prefs.compute_device_type = device_type
            cycles_prefs.refresh_devices()
            enabled = []
            for d in cycles_prefs.devices:
                if d.type == device_type:
                    d.use = True
                    enabled.append(d)
            if enabled:
                print(f"Cycles device: GPU ({device_type}) — {len(enabled)} device(s): {[x.name for x in enabled]}")
                return
        except Exception as e:
            print(f"Cycles {device_type} failed: {e}")
            continue
    scene.cycles.device = "CPU"
    print("Cycles device: CPU (no GPU enabled).")
    # Quick diagnostic: can the system see the GPU from this process?
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            print("  nvidia-smi sees GPU(s):", r.stdout.strip()[:80] + ("..." if len(r.stdout) > 80 else ""))
            print("  → CUDA cuInit failed: Blender 3.6 was built for older CUDA; driver 590 (CUDA 13) is too new. Fix: use Blender 4.2+ from blender.org (built for newer drivers) or enable OptiX in Blender GUI (Edit > Preferences > System > Cycles) and Save.")
        else:
            print("  nvidia-smi failed or no GPU. Install/update NVIDIA driver and run: nvidia-smi")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        print("  Tip: In Blender GUI use Edit > Preferences > System > Cycles to enable GPU and save. If CUDA cuInit persists, run 'nvidia-smi' and check driver/libcuda.")


def render_tree(scene):
    scene.render.engine = "CYCLES"
    set_cycles_gpu(scene, prefer_optix=True)
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False


def _add_depth_output_node(scene, depth_dir, depth_prefix):
    """Add a depth File Output node to the existing compositor tree.

    Must be called AFTER setup_mask_tree_only / setup_mask_tree_and_box
    (which build the compositor tree with a RenderLayers node).
    Writes a 32-bit float EXR with the depth pass.
    """
    nt = scene.node_tree
    # Fast path: depth node already exists — just update paths.
    out_depth = nt.nodes.get("DepthFileOutput")
    if out_depth is not None:
        out_depth.base_path = depth_dir
        out_depth.file_slots[0].path = f"{depth_prefix}_z_"
        return
    # Find the existing RenderLayers node
    rl = None
    for node in nt.nodes:
        if node.type == "R_LAYERS":
            rl = node
            break
    if rl is None:
        print("  Warning: no RenderLayers node found, skipping depth output node")
        return
    out_depth = nt.nodes.new("CompositorNodeOutputFile")
    out_depth.name = "DepthFileOutput"
    out_depth.base_path = depth_dir
    out_depth.format.file_format = "OPEN_EXR"
    out_depth.format.color_depth = "32"
    out_depth.format.color_mode = "RGB"
    out_depth.file_slots[0].path = f"{depth_prefix}_z_"
    out_depth.location = (200, -200)
    nt.links.new(rl.outputs["Depth"], out_depth.inputs[0])


def save_depth_npy(scene, depth_dir, depth_prefix, depth_npy_path):
    """Load the depth EXR written by the compositor, convert to .npy, and clean up.

    The compositor File Output node writes: <depth_dir>/<depth_prefix>_z_NNNN.exr
    """
    import glob as _g
    pattern = os.path.join(depth_dir, f"{depth_prefix}_z_*.exr")
    matches = _g.glob(pattern)
    if not matches:
        print(f"  Warning: no depth EXR found matching {pattern}")
        return
    exr_path = matches[0]
    # Load EXR into Blender, extract pixels, save as .npy
    img = bpy.data.images.load(exr_path)
    w, h = img.size[0], img.size[1]
    n_channels = img.channels
    pixels = np.empty(w * h * n_channels, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    depth_map = pixels.reshape(h, w, n_channels)[::-1, :, 0]
    np.save(depth_npy_path, depth_map)
    # Clean up: remove temp EXR and Blender image datablock
    bpy.data.images.remove(img)
    try:
        os.remove(exr_path)
    except OSError:
        pass
    print(f"  Saved depth: {depth_npy_path}  shape={depth_map.shape}  "
          f"range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")


def write_annotation(ann_path, tree_id, shot_idx, variant, cam_loc, cam_rot, rgb_path, depth_path, mask_path, K, background_objects, tree_obj_name, cylinders_world):
    ann = {
        "tree_id": tree_id,
        "shot": shot_idx,
        "variant": variant,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "masks": {"tree_only": mask_path, "union": None},
        "camera": {
            "location": list(cam_loc),
            "rotation_euler": list(cam_rot),
            "intrinsics": {"width": W, "height": H, "K": K},
        },
        "reference": {"post0": POST0_NAME, "post1": POST1_NAME},
        "tree_object": tree_obj_name,
        "background_objects": background_objects,
    }
    ann["cylinders_world"] = cylinders_world
    with open(ann_path, "w") as f:
        json.dump(ann, f, indent=2)


def update_camera_rect(cam_obj):
    """
    Create or update a dark-gray rectangle that is always visible in the camera frame.

    Placed at a fixed depth (CAMERA_RECT_DEPTH) in front of the camera, perpendicular
    to the view direction, and vertically offset by CAMERA_RECT_FRAME_V_OFFSET so it
    sits in the lower portion of the frame.  Because placement is in camera space the
    rect follows the camera exactly regardless of camera position or Z level.
    """
    RECT_NAME = "camera_ground_rect"
    existing = bpy.data.objects.get(RECT_NAME)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)

    if not ENABLE_CAMERA_RECT:
        return None

    mw      = cam_obj.matrix_world
    cam_loc = mw.to_translation()

    # Camera axes in world space (Blender: camera looks along local -Z)
    # matrix_world columns: col0 = right, col1 = up, col2 = back → forward = -col2
    fwd   = mathutils.Vector((-mw[0][2], -mw[1][2], -mw[2][2])).normalized()
    right = mathutils.Vector(( mw[0][0],  mw[1][0],  mw[2][0])).normalized()
    up    = mathutils.Vector(( mw[0][1],  mw[1][1],  mw[2][1])).normalized()

    depth = CAMERA_RECT_DEPTH

    # Map frame offsets (-1..1) to world units at this depth using intrinsics.
    # K[1][1]=fy → half-height in world at depth d = d*(H/2)/fy
    # K[0][0]=fx → half-width  in world at depth d = d*(W/2)/fx
    half_h_world   = depth * (H / 2.0) / K_REF[1][1]
    half_w_world   = depth * (W / 2.0) / K_REF[0][0]
    v_offset_world = CAMERA_RECT_FRAME_V_OFFSET * half_h_world
    h_offset_world = CAMERA_RECT_FRAME_H_OFFSET * half_w_world

    # Center of the rectangle: fixed depth along view + shifts in camera space
    center = cam_loc + depth * fwd + v_offset_world * up + h_offset_world * right

    half_rect_h = CAMERA_RECT_HEIGHT_M  / 2.0
    half_rect_w = CAMERA_RECT_WIDTH_M   / 2.0
    thickness   = CAMERA_RECT_THICKNESS_M

    # 8 corners of a box:
    #   Front face (0-3): at `center`, facing the camera.
    #   Back  face (4-7): shifted by thickness along fwd (away from camera).
    #   Longer dimension (HEIGHT_M) goes along right → landscape/sideways orientation.
    t = fwd * thickness
    f0 = center - half_rect_h * right - half_rect_w * up
    f1 = center + half_rect_h * right - half_rect_w * up
    f2 = center + half_rect_h * right + half_rect_w * up
    f3 = center - half_rect_h * right + half_rect_w * up
    b0, b1, b2, b3 = f0 + t, f1 + t, f2 + t, f3 + t

    verts_list = [(v.x, v.y, v.z) for v in (f0, f1, f2, f3, b0, b1, b2, b3)]
    faces = [
        (0, 1, 2, 3),  # front  (faces camera)
        (5, 4, 7, 6),  # back
        (0, 4, 5, 1),  # bottom
        (3, 2, 6, 7),  # top
        (0, 3, 7, 4),  # left
        (1, 5, 6, 2),  # right
    ]

    print(f"  [camera_rect] depth={depth:.2f}m  v={v_offset_world:.3f}m  h={h_offset_world:.3f}m  "
          f"thickness={thickness:.3f}m  center=({center.x:.3f},{center.y:.3f},{center.z:.3f})")

    mesh = bpy.data.meshes.new(name=RECT_NAME + "_mesh")
    mesh.from_pydata(verts_list, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(RECT_NAME, mesh)
    bpy.context.collection.objects.link(obj)

    # Metallic steel material with procedural noise for surface variation
    mat_name = "MetallicRect"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.use_backface_culling = False
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Noise texture for subtle surface colour variation (brushed-metal look)
        noise = nodes.new("ShaderNodeTexNoise")
        noise.location = (-500, 0)
        noise.inputs["Scale"].default_value    = 120.0
        noise.inputs["Detail"].default_value   = 8.0
        noise.inputs["Roughness"].default_value = 0.55
        noise.inputs["Distortion"].default_value = 0.1

        # Remap noise output to a narrow steel-gray range
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-280, 0)
        ramp.color_ramp.interpolation = "LINEAR"
        ramp.color_ramp.elements[0].position = 0.4
        ramp.color_ramp.elements[0].color = (0.45, 0.47, 0.50, 1.0)  # dark steel
        ramp.color_ramp.elements[1].position = 0.65
        ramp.color_ramp.elements[1].color = (0.80, 0.82, 0.85, 1.0)  # light steel

        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (200, 0)
        principled.inputs["Metallic"].default_value   = 1.0
        principled.inputs["Roughness"].default_value  = 0.25

        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (480, 0)

        links.new(noise.outputs["Fac"],   ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"],  principled.inputs["Base Color"])
        links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)

    obj.pass_index      = PASS_INDEX_BOX          # isolated pass so box mask can be generated
    obj.hide_viewport   = False
    obj.hide_render     = False
    bpy.context.view_layer.update()
    return obj


def render_tree_2(scene, cam_obj, tree_id, tree_obj, z_levels, K, texture_name, background_tree_obj=None):
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    # Main tree and optional background tree visible; others hidden (use names so both are in render)
    main_name = tree_obj.name
    for obj in bpy.data.objects:
        if obj.name.startswith("tree") and "_TRUNK" in obj.name:
            visible = obj.name == main_name or (
                background_tree_obj is not None and obj.name == TREE_BG_OBJ
            )
            obj.hide_viewport = not visible
            obj.hide_render = not visible
    if background_tree_obj is not None:
        background_tree_obj.hide_render = False
        background_tree_obj.hide_viewport = False
    bpy.context.view_layer.update()
    print(f"  Rendering tree object: {tree_obj.name} (materials: {[m.name for m in tree_obj.data.materials]})")
    background_objects = get_background_objects()
    set_pass_indices(tree_obj, background_objects, background_tree_obj)
    rot_ref = ROTATION_REF

    # ---- Load + convert cylinders ONCE per tree (centroid + orientation to world) ----
    meta_path = os.path.join(METADATA_DIR, f"{tree_id}_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cylinders_local = get_cylinder_data_from_metadata(meta)
    cyl_world_full = transform_cylinders_world(tree_obj, cylinders_local)
    # For annotations: list of [x,y,z] centroids only (backward compatible)
    cyl_world = [c["centroid"] for c in cyl_world_full]
    # Store one cylinders_world file per tree: centroid + orientation (and radius, length, part_name)
    cylinders_dir = os.path.join(CYLINDERS_WORLD_DIR, texture_name)
    os.makedirs(cylinders_dir, exist_ok=True)
    cyl_file = os.path.join(cylinders_dir, f"{tree_id}.json")
    with open(cyl_file, "w") as f:
        json.dump(cyl_world_full, f, indent=2)
    print(f"  Saved cylinders_world: {cyl_file} ({len(cyl_world_full)} cylinders, centroid + orientation)")

    for shot_idx, z_val in enumerate(z_levels, start=1):
        loc_center = [X_REF, Y_REF, z_val]
        cam_obj.location = loc_center
        cam_obj.rotation_mode = "XYZ"
        cam_obj.rotation_euler = rot_ref
        bpy.context.view_layer.update()

        # render_tree_2 always renders WITHOUT the rectangle (box_cam is handled by semicircle)
        _rect = bpy.data.objects.get("camera_ground_rect")
        if _rect:
            bpy.data.objects.remove(_rect, do_unlink=True)

        subdir = "box"
        rgb_dir   = os.path.join(RGB_DIR_4,   texture_name, tree_id, subdir)
        depth_dir = os.path.join(DEPTH_DIR_4, texture_name, tree_id, subdir)
        ann_dir   = os.path.join(ANN_DIR_4,   texture_name, tree_id, subdir)
        mask_dir  = os.path.join(MASK_DIR_4,  texture_name, tree_id, subdir)
        os.makedirs(rgb_dir,   exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(ann_dir,   exist_ok=True)
        os.makedirs(mask_dir,  exist_ok=True)

        # ---- Center: rgb_4, depth_4, ann_4, mask_4 (_c) ----
        prefix_c = f"{tree_id}_shot{shot_idx:02d}_c"
        setup_mask_tree_only(scene, mask_dir, prefix_c)
        depth_prefix_c = f"{tree_id}_shot{shot_idx:02d}"
        _add_depth_output_node(scene, depth_dir, depth_prefix_c)
        scene.frame_set(shot_idx)

        rgb_name = f"{tree_id}_shot{shot_idx:02d}.png"
        depth_name = f"{tree_id}_shot{shot_idx:02d}.npy"
        ann_name = f"{tree_id}_shot{shot_idx:02d}.json"
        rgb_path = os.path.join(rgb_dir, rgb_name)
        depth_path = os.path.join(depth_dir, depth_name)
        ann_path = os.path.join(ann_dir, ann_name)

        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)

        # Convert depth EXR (written by compositor) to .npy and clean up
        save_depth_npy(scene, depth_dir, depth_prefix_c, depth_path)

        # Rename mask to remove Blender frame-number suffix (_tree_0001 → clean name)
        rename_mask_output(mask_dir, prefix_c)
        mask_tree_path = os.path.join(mask_dir, f"{prefix_c}.png")

        loc = cam_obj.matrix_world.to_translation()
        eul = cam_obj.matrix_world.to_euler("XYZ")

        write_annotation(
            ann_path, tree_id, shot_idx, "c",
            (loc.x, loc.y, loc.z), (eul.x, eul.y, eul.z),
            rgb_path, depth_path, mask_tree_path, K,
            background_objects, tree_obj.name,
            cyl_world
        )

        bpy.context.view_layer.update()
        print(f"  ✓ shot{shot_idx:02d} center: {rgb_name}, depth, mask_4 ({prefix_c})")

        # ---- Optical flow (left/right) ----
        if ENABLE_OPTICAL_FLOW:
            flow_dir = os.path.join(OPTICAL_FLOW_DIR_4, texture_name, tree_id, subdir)
            os.makedirs(flow_dir, exist_ok=True)

            # Camera right vector: Blender matrix_world columns 0,1,2 = right, up, back in world
            mw = cam_obj.matrix_world
            right = (mw[0][0], mw[1][0], mw[2][0])
            dx = DX_OFFSET

            # ---- Right: Optical_flow_4 _r, depth _r, mask_4 _r ----
            loc_r = [loc_center[i] + dx * right[i] for i in range(3)]
            cam_obj.location = loc_r
            bpy.context.view_layer.update()
            prefix_r = f"{tree_id}_shot{shot_idx:02d}_r"
            setup_mask_tree_only(scene, mask_dir, prefix_r)
            flow_r_path = os.path.join(flow_dir, f"{tree_id}_shot{shot_idx:02d}_r.png")
            depth_r_path = os.path.join(depth_dir, f"{tree_id}_shot{shot_idx:02d}_r.npy")
            ann_r_path = os.path.join(ann_dir, f"{tree_id}_shot{shot_idx:02d}_r.json")
            _add_depth_output_node(scene, depth_dir, prefix_r)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.filepath = flow_r_path
            bpy.ops.render.render(write_still=True)
            save_depth_npy(scene, depth_dir, prefix_r, depth_r_path)
            rename_mask_output(mask_dir, prefix_r)
            loc = cam_obj.matrix_world.to_translation()
            eul = cam_obj.matrix_world.to_euler("XYZ")
            write_annotation(
                ann_r_path, tree_id, shot_idx, "r",
                (loc.x, loc.y, loc.z), (eul.x, eul.y, eul.z),
                flow_r_path, depth_r_path, os.path.join(mask_dir, f"{prefix_r}.png"),
                K, background_objects, tree_obj.name,
                cyl_world,
            )
            print(f"  ✓ shot{shot_idx:02d} right: Optical_flow_4 _r, depth _r, mask_4 _r")

            # ---- Left: Optical_flow_4 _l, depth _l, mask_4 _l ----
            loc_l = [loc_center[i] - dx * right[i] for i in range(3)]
            cam_obj.location = loc_l
            bpy.context.view_layer.update()
            prefix_l = f"{tree_id}_shot{shot_idx:02d}_l"
            setup_mask_tree_only(scene, mask_dir, prefix_l)
            flow_l_path = os.path.join(flow_dir, f"{tree_id}_shot{shot_idx:02d}_l.png")
            depth_l_path = os.path.join(depth_dir, f"{tree_id}_shot{shot_idx:02d}_l.npy")
            ann_l_path = os.path.join(ann_dir, f"{tree_id}_shot{shot_idx:02d}_l.json")
            _add_depth_output_node(scene, depth_dir, prefix_l)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.filepath = flow_l_path
            bpy.ops.render.render(write_still=True)
            save_depth_npy(scene, depth_dir, prefix_l, depth_l_path)
            rename_mask_output(mask_dir, prefix_l)
            loc = cam_obj.matrix_world.to_translation()
            eul = cam_obj.matrix_world.to_euler("XYZ")
            write_annotation(
                ann_l_path, tree_id, shot_idx, "l",
                (loc.x, loc.y, loc.z), (eul.x, eul.y, eul.z),
                flow_l_path, depth_l_path, os.path.join(mask_dir, f"{prefix_l}.png"),
                K, background_objects, tree_obj.name,
                cyl_world,
            )
            print(f"  ✓ shot{shot_idx:02d} left: Optical_flow_4 _l, depth _l, mask_4 _l")

    n = len(z_levels)
    flow_msg = f", Optical_flow_4 ({n*2} l+r), depth ({n*2} l+r)" if ENABLE_OPTICAL_FLOW else ""
    print(f"  Done {tree_id} [{texture_name}]: rgb_4 ({n} center), depth_4 ({n}), ann_4 ({n}), mask_4 ({n}){flow_msg}")


def render_semicircle_cameras(scene, cam_obj, tree_id, tree_obj, K,
                              texture_name, cyl_json_path, metadata_path,
                              background_tree_obj=None, cam_prefix="cam",
                              num_poses=None, with_camera_rect=False):
    """
    Render from camera poses along an arc centered on the reference camera.
    Each angular position renders NUM_Z_LEVELS shots going up the Z axis.

    Args:
        num_poses: override for number of angular positions (defaults to NUM_SEMICIRCLE_POSES).
        with_camera_rect: if True, place the camera rectangle in every shot.
    """
    from move_camera import generate_semicircle_poses

    if num_poses is None:
        num_poses = NUM_SEMICIRCLE_POSES

    # Get the tree-specific trunk part name from metadata (e.g. trunk_1, trunk_3, ...)
    trunk_part_name = get_root_name(metadata_path) or "trunk_1"

    poses = generate_semicircle_poses(
        cylinder_json_path=cyl_json_path,
        x_ref=X_REF, y_ref=Y_REF, z_ref=Z_REF, z_final=Z_FINAL,
        rotation_ref=ROTATION_REF,
        trunk_part_name=trunk_part_name,
        num_poses=num_poses,
        num_z_levels=NUM_Z_LEVELS,
        central_angle=SEMICIRCLE_CENTRAL_ANGLE,
    )

    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True

    # Visibility: same logic as render_tree_2
    main_name = tree_obj.name
    for obj in bpy.data.objects:
        if obj.name.startswith("tree") and "_TRUNK" in obj.name:
            visible = obj.name == main_name or (
                background_tree_obj is not None and obj.name == TREE_BG_OBJ
            )
            obj.hide_viewport = not visible
            obj.hide_render = not visible
    if background_tree_obj is not None:
        background_tree_obj.hide_render = False
        background_tree_obj.hide_viewport = False
    bpy.context.view_layer.update()

    # Remove camera rect unless this pass needs it
    if not with_camera_rect:
        _rect = bpy.data.objects.get("camera_ground_rect")
        if _rect:
            bpy.data.objects.remove(_rect, do_unlink=True)
        for m in list(bpy.data.meshes):
            if m.name.startswith("camera_ground_rect") and m.users == 0:
                bpy.data.meshes.remove(m)
        bpy.context.view_layer.update()

    background_objects = get_background_objects()
    set_pass_indices(tree_obj, background_objects, background_tree_obj)

    # Load cylinders_world for annotations
    meta_path = os.path.join(METADATA_DIR, f"{tree_id}_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cylinders_local = get_cylinder_data_from_metadata(meta)
    cyl_world_full = transform_cylinders_world(tree_obj, cylinders_local)
    cyl_world = [c["centroid"] for c in cyl_world_full]

    frame_counter = 0
    for pose in poses:
        cam_idx = pose["cam_idx"]
        shot_idx = pose["shot_idx"]
        cam_dir = f"{cam_prefix}{cam_idx}"

        # Create per-camera output directories
        rgb_dir = os.path.join(RGB_DIR_4, texture_name, tree_id, cam_dir)
        depth_dir = os.path.join(DEPTH_DIR_4, texture_name, tree_id, cam_dir)
        ann_dir = os.path.join(ANN_DIR_4, texture_name, tree_id, cam_dir)
        mask_dir = os.path.join(MASK_DIR_4, texture_name, tree_id, cam_dir)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # Set camera pose
        cam_obj.location = pose["location"]
        cam_obj.rotation_mode = "XYZ"
        cam_obj.rotation_euler = pose["rotation_euler"]
        bpy.context.view_layer.update()

        if with_camera_rect:
            update_camera_rect(cam_obj)
        else:
            # Ensure no rect in scene
            _rect = bpy.data.objects.get("camera_ground_rect")
            if _rect:
                bpy.data.objects.remove(_rect, do_unlink=True)

        shot_label = f"shot{shot_idx:02d}"
        prefix = f"{tree_id}_{shot_label}"
        if with_camera_rect:
            box_mask_dir = os.path.join(BOX_MASK_DIR, texture_name, tree_id, cam_dir)
            os.makedirs(box_mask_dir, exist_ok=True)
            setup_mask_tree_and_box(scene, mask_dir, prefix, box_mask_dir, prefix)
        else:
            setup_mask_tree_only(scene, mask_dir, prefix)
        _add_depth_output_node(scene, depth_dir, prefix)
        frame_counter += 1
        scene.frame_set(frame_counter)

        rgb_name = f"{tree_id}_{shot_label}.png"
        depth_name = f"{tree_id}_{shot_label}.npy"
        ann_name = f"{tree_id}_{shot_label}.json"
        rgb_path = os.path.join(rgb_dir, rgb_name)
        depth_path = os.path.join(depth_dir, depth_name)
        ann_path = os.path.join(ann_dir, ann_name)
        # Render RGB
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)

        # Convert depth EXR (written by compositor) to .npy and clean up
        save_depth_npy(scene, depth_dir, prefix, depth_path)

        # Rename mask to remove Blender frame-number suffix
        rename_mask_output(mask_dir, prefix)
        mask_tree_path = os.path.join(mask_dir, f"{prefix}.png")

        # Write annotation
        loc = cam_obj.matrix_world.to_translation()
        eul = cam_obj.matrix_world.to_euler("XYZ")
        write_annotation(
            ann_path, tree_id, shot_idx, shot_label,
            (loc.x, loc.y, loc.z), (eul.x, eul.y, eul.z),
            rgb_path, depth_path, mask_tree_path, K,
            background_objects, tree_obj.name,
            cyl_world,
        )

        bpy.context.view_layer.update()
        print(f"    {cam_dir}/{shot_label}: {rgb_name}, depth, mask, ann")

        # ---- Optical flow (left/right) for this semicircle pose ----
        if ENABLE_OPTICAL_FLOW:
            flow_dir = os.path.join(OPTICAL_FLOW_DIR_4, texture_name, tree_id, cam_dir)
            os.makedirs(flow_dir, exist_ok=True)

            loc_center_flow = list(pose["location"])
            mw = cam_obj.matrix_world
            right_vec = (mw[0][0], mw[1][0], mw[2][0])
            dx = DX_OFFSET

            # ---- Right ----
            loc_r = [loc_center_flow[i] + dx * right_vec[i] for i in range(3)]
            cam_obj.location = loc_r
            bpy.context.view_layer.update()
            prefix_r = f"{tree_id}_{shot_label}_r"
            if with_camera_rect:
                box_mask_dir = os.path.join(BOX_MASK_DIR, texture_name, tree_id, cam_dir)
                os.makedirs(box_mask_dir, exist_ok=True)
                setup_mask_tree_and_box(scene, mask_dir, prefix_r, box_mask_dir, prefix_r)
            else:
                setup_mask_tree_only(scene, mask_dir, prefix_r)
            flow_r_path = os.path.join(flow_dir, f"{tree_id}_{shot_label}_r.png")
            depth_r_path = os.path.join(depth_dir, f"{tree_id}_{shot_label}_r.npy")
            ann_r_path = os.path.join(ann_dir, f"{tree_id}_{shot_label}_r.json")
            _add_depth_output_node(scene, depth_dir, prefix_r)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.filepath = flow_r_path
            bpy.ops.render.render(write_still=True)
            save_depth_npy(scene, depth_dir, prefix_r, depth_r_path)
            rename_mask_output(mask_dir, prefix_r)
            loc_obj = cam_obj.matrix_world.to_translation()
            eul_obj = cam_obj.matrix_world.to_euler("XYZ")
            write_annotation(
                ann_r_path, tree_id, shot_idx, f"{shot_label}_r",
                (loc_obj.x, loc_obj.y, loc_obj.z), (eul_obj.x, eul_obj.y, eul_obj.z),
                flow_r_path, depth_r_path, os.path.join(mask_dir, f"{prefix_r}.png"),
                K, background_objects, tree_obj.name, cyl_world,
            )
            print(f"    {cam_dir}/{shot_label}_r: optical flow right, depth, mask")

            # ---- Left ----
            loc_l = [loc_center_flow[i] - dx * right_vec[i] for i in range(3)]
            cam_obj.location = loc_l
            bpy.context.view_layer.update()
            prefix_l = f"{tree_id}_{shot_label}_l"
            if with_camera_rect:
                box_mask_dir = os.path.join(BOX_MASK_DIR, texture_name, tree_id, cam_dir)
                os.makedirs(box_mask_dir, exist_ok=True)
                setup_mask_tree_and_box(scene, mask_dir, prefix_l, box_mask_dir, prefix_l)
            else:
                setup_mask_tree_only(scene, mask_dir, prefix_l)
            flow_l_path = os.path.join(flow_dir, f"{tree_id}_{shot_label}_l.png")
            depth_l_path = os.path.join(depth_dir, f"{tree_id}_{shot_label}_l.npy")
            ann_l_path = os.path.join(ann_dir, f"{tree_id}_{shot_label}_l.json")
            _add_depth_output_node(scene, depth_dir, prefix_l)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.filepath = flow_l_path
            bpy.ops.render.render(write_still=True)
            save_depth_npy(scene, depth_dir, prefix_l, depth_l_path)
            rename_mask_output(mask_dir, prefix_l)
            loc_obj = cam_obj.matrix_world.to_translation()
            eul_obj = cam_obj.matrix_world.to_euler("XYZ")
            write_annotation(
                ann_l_path, tree_id, shot_idx, f"{shot_label}_l",
                (loc_obj.x, loc_obj.y, loc_obj.z), (eul_obj.x, eul_obj.y, eul_obj.z),
                flow_l_path, depth_l_path, os.path.join(mask_dir, f"{prefix_l}.png"),
                K, background_objects, tree_obj.name, cyl_world,
            )
            print(f"    {cam_dir}/{shot_label}_l: optical flow left, depth, mask")

            # Restore camera to center pose for next iteration
            cam_obj.location = loc_center_flow
            cam_obj.rotation_euler = pose["rotation_euler"]
            bpy.context.view_layer.update()

    n_cams = num_poses
    n_shots = NUM_Z_LEVELS
    flow_msg = f" + {n_cams * n_shots * 2} optical flow" if ENABLE_OPTICAL_FLOW else ""
    print(f"  Done {tree_id} [{texture_name}]: semi-circle ({n_cams} cameras x {n_shots} Z-levels = {n_cams * n_shots} center{flow_msg} renders)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RGB_DIR_4, exist_ok=True)
    os.makedirs(DEPTH_DIR_4, exist_ok=True)
    os.makedirs(ANN_DIR_4, exist_ok=True)
    os.makedirs(MASK_DIR_4, exist_ok=True)
    os.makedirs(BOX_MASK_DIR, exist_ok=True)
    os.makedirs(OPTICAL_FLOW_DIR_4, exist_ok=True)
    os.makedirs(CYLINDERS_WORLD_DIR, exist_ok=True)
    os.makedirs(TEXTURES_DIR, exist_ok=True)

    # Per-task selection from SLURM array (set by generate_tree.sh).
    # If not set, processes all textures and trees (useful for local/debug runs).
    _task_bark = os.environ.get("BARK_NAME")
    _task_tree = os.environ.get("TREE_ID")

    texture_entries = BARK_TEXTURES
    if _task_bark:
        texture_entries = [t for t in BARK_TEXTURES if t["name"] == _task_bark]
        if not texture_entries:
            print(f"ERROR: BARK_NAME='{_task_bark}' not found in BARK_TEXTURES. Exiting.")
            return

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    # Remove blend template placeholders (Cylinder.*, Cube) that show purple when textures are missing on HPC
    remove_placeholder_objects()
    # Replace world background with solid color (blend HDR/image paths break on HPC → whole frame purple)
    fix_world_background()
    if LOW_LIGHT_MODE:
        setup_low_light_sun()
    # Replace ground material with dirt_floor from TEXTURES_DIR (blend paths from local machine → purple on HPC)
    fix_ground_material()
    cam_obj = bpy.data.objects.get(CAM)
    if cam_obj is None:
        raise RuntimeError(f"Camera '{CAM}' not found")
    cam_obj.data.lens = 28.0
    scene.camera = cam_obj
    render_tree(scene)

    z_levels = get_z_levels()
    print(f"Z levels ({NUM_Z_LEVELS}): {z_levels}")
    print(f"Bark texture sets: {[t['name'] for t in texture_entries]}")
    print(f"Textures base dir (absolute): {_blender_path(TEXTURES_DIR)}")
    # Resolve first set so user can verify paths
    d0, n0 = find_texture_paths(BARK_TEXTURES[0])
    print(f"First set '{BARK_TEXTURES[0]['name']}': diff={d0}, normal={n0}")
    print(f"  (diff exists: {os.path.isfile(d0) if d0 else False}, normal exists: {os.path.isfile(n0) if n0 else False})")
    K = K_REF

    all_metadata_files = find_all_metadata_files()
    if TREE_ID_FILTER:
        all_metadata_files = [(tid, path, num) for tid, path, num in all_metadata_files if TREE_ID_FILTER in tid.lower()]
        print(f"Filtered to trees with '{TREE_ID_FILTER}' in name: {len(all_metadata_files)} trees")
    metadata_files = all_metadata_files
    if _task_tree:
        metadata_files = [(tid, path, num) for tid, path, num in all_metadata_files if tid == _task_tree]
        if not metadata_files:
            print(f"[SKIP] No metadata found for TREE_ID='{_task_tree}'.")
            return
        print(f"Selected tree '{_task_tree}'")
    if not metadata_files:
        print("No metadata found.")
        return

    saved_tree_data = {}
    for idx, (tree_id, _, _) in enumerate(metadata_files):
        obj_name = TREE_OBJ.format(idx)
        obj = bpy.data.objects.get(obj_name)
        if obj:
            loc = obj.matrix_world.to_translation().copy()
            mat = obj.data.materials[0] if obj.data.materials else None
            saved_tree_data[idx] = (loc, mat)

    remove_all_tree_objects()

    num_trees = len(metadata_files)
    for texture_entry in texture_entries:
        texture_name = texture_entry["name"]
        print(f"\n=== Texture: {texture_name} ({texture_entry['folder']}/{texture_entry['prefix']}) ===")
        for tree_idx, (tree_id, metadata_path, file_num) in enumerate(metadata_files):
            print(f"\n[{tree_idx + 1}/{num_trees}] Tree: {tree_id}")
            # Skip if already rendered (resume support)
            _done_marker = os.path.join(RGB_DIR_4, texture_name, tree_id)
            if not FORCE_RENDER and os.path.isdir(_done_marker) and len(os.listdir(_done_marker)) > 0:
                print(f"  [SKIP] Already rendered: {_done_marker}")
                continue
            tree_obj = get_tree_object_for_metadata(
                tree_id=tree_id,
                metadata_path=metadata_path,
                tree_idx=tree_idx,
                saved_tree_data=saved_tree_data,
                texture_entry=texture_entry,
            )
            if tree_obj is None:
                print(f"  Skipping {tree_id}")
                continue
            # One random other tree behind and left/right (pick from full tree list)
            bg_obj = None
            if ENABLE_BACKGROUND_TREE and len(all_metadata_files) >= 2:
                bg_candidates = [tid for tid, _, _ in all_metadata_files if tid != tree_id]
                tree_id_bg = random.choice(bg_candidates)
                bg_obj = load_background_tree(tree_id_bg, tree_obj, texture_entry)
                if bg_obj:
                    print(f"  Background tree: {tree_id_bg} (behind + lateral)")
            if not BOXCAM_ONLY:
                render_tree_2(
                    scene=scene,
                    cam_obj=cam_obj,
                    tree_id=tree_id,
                    tree_obj=tree_obj,
                    z_levels=z_levels,
                    K=K,
                    texture_name=texture_name,
                    background_tree_obj=bg_obj,
                )

            # Semi-circular cameras: render from 5 angular positions around the trunk
            if ENABLE_SEMICIRCLE_CAMERAS:
                cyl_json = os.path.join(
                    CYLINDERS_WORLD_DIR, texture_name, f"{tree_id}.json"
                )
                if os.path.isfile(cyl_json):
                    # Normal lighting: 10 cam shots → cam1/ ... cam10/
                    if not BOXCAM_ONLY:
                        render_semicircle_cameras(
                            scene=scene,
                            cam_obj=cam_obj,
                            tree_id=tree_id,
                            tree_obj=tree_obj,
                            K=K,
                            texture_name=texture_name,
                            cyl_json_path=cyl_json,
                            metadata_path=metadata_path,
                            background_tree_obj=bg_obj,
                            cam_prefix="cam",
                            num_poses=NUM_SEMICIRCLE_POSES,
                            with_camera_rect=False,
                        )

                    # Box-cam: 8 cam shots WITH rectangle → box_cam1/ ... box_cam8/
                    if ENABLE_CAMERA_RECT:
                        render_semicircle_cameras(
                            scene=scene,
                            cam_obj=cam_obj,
                            tree_id=tree_id,
                            tree_obj=tree_obj,
                            K=K,
                            texture_name=texture_name,
                            cyl_json_path=cyl_json,
                            metadata_path=metadata_path,
                            background_tree_obj=bg_obj,
                            cam_prefix="box_cam",
                            num_poses=NUM_BOX_CAM_POSES,
                            with_camera_rect=True,
                        )

                    if ENABLE_DARK_LIGHT_CAM:
                        # Dark lighting: 10 cam shots → dark_light_cam1/ ... dark_light_cam10/
                        print(f"  Switching to dark lighting for {tree_id}...")
                        fix_world_background_dark()
                        setup_low_light_sun()
                        render_semicircle_cameras(
                            scene=scene,
                            cam_obj=cam_obj,
                            tree_id=tree_id,
                            tree_obj=tree_obj,
                            K=K,
                            texture_name=texture_name,
                            cyl_json_path=cyl_json,
                            metadata_path=metadata_path,
                            background_tree_obj=bg_obj,
                            cam_prefix="dark_light_cam",
                            num_poses=NUM_SEMICIRCLE_POSES,
                            with_camera_rect=False,
                        )
                        # Restore normal lighting for next tree
                        print(f"  Restoring normal lighting...")
                        fix_world_background()
                        remove_low_light_sun()
                else:
                    print(f"  Skipping semi-circle cameras: {cyl_json} not found")

    print(f"\n✓ generate_tree2 done. Bark: {_task_bark or 'all'}, Tree: {_task_tree or 'all'}.")


if __name__ == "__main__":
    main()
