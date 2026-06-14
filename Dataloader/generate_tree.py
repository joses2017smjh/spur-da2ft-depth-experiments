import bpy
import os 
import json
import math
import random
import re



SCRIPT_DIR = "/home/joses/Computer_Vision/trees"
CONSTANT_CAMERA = "/home/joses/Computer_Vision/toy_example/ann"

REFERENCE_ANNOTATIONS = [
    os.path.join(CONSTANT_CAMERA, "frame_0009.json"),
    os.path.join(CONSTANT_CAMERA, "frame_0016.json"),
    os.path.join(CONSTANT_CAMERA, "frame_0023.json"),
]

METADATA_DIR = "/home/joses/Computer_Vision/trees/metadata"
PLY_DIR = "/home/joses/Computer_Vision/trees/ply"

OUTPUT_DIR = "/home/joses/Computer_Vision/Data"
CAM = "Camera"
TREE_OBJ = "tree{}_TRUNK"

W, H = 1920, 1080

VARIATIONS_SHOT = 20

POST0_NAME = "post0"
POST1_NAME = "post1"

RGB_DIR = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")
ANN_DIR = os.path.join(OUTPUT_DIR, "ann")
MASK_DIR = os.path.join(OUTPUT_DIR, "mask")

# Pass indices for segmentation
PASS_INDEX_TREE = 1
PASS_INDEX_BACKGROUND = 2  # posts, wires, ground


""" Gets tae trunk root name from the metadata file """
def get_root_name(meta_pth):
    try:
        with open(meta_pth, 'r') as f:
            metadata = json.load(f)
        hierarchy = metadata.get('hierarchy', {})
        root = hierarchy.get('root', [])
        if root and len(root) > 0:
            return root[0]
    except Exception as e:
        print(f"error reading metadata {meta_pth}: {e}")
    return None

""" Extracts all the json files with tree IDs """
def find_all_metadata_files(): 
    if not os.path.exists(METADATA_DIR):
        return []

    metadata_files = []
    for fname in sorted(os.listdir(METADATA_DIR)):
            if fname.endswith("_metadata.json"):
                tree_id = fname.replace("_metadata.json", "")
                metadata_path = os.path.join(METADATA_DIR, fname)
                match = re.search(r'(\d+)$', tree_id)
                file_num = int(match.group(1)) if match else 0
                metadata_files.append((tree_id, metadata_path, file_num))
    return metadata_files

""" Loods the PLY file into Blender and returns the imported object """
def load_ply_into_blender(ply_pth, object_name):
    if not os.path.exists(ply_pth):
        print(f"PLY does not exits: {ply_pth}")
        return None
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_mesh.ply(filepath=ply_pth)

    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print(f"Import failed for {ply_pth}")
        return None

    #allows the object to be visible in the depth maps and when rendering camera shots
    obj.name = object_name
    obj.hide_viewport = False
    obj.hide_render  = False

    print(f"the ply has been loaded as {os.path.basename(ply_pth)} as {object_name}")
    return obj

""" removes all the tree object in the scene """
def remove_all_tree_objects():  
    objects_to_remove = []
    for obj in bpy.data.objects:
        if(obj.name.startswith("tree")) and ("_TRUNK" in obj.name or "_BRANCH" in obj.name or "_SPUR" in obj.name):
            objects_to_remove.append(obj)
    
    if objects_to_remove:
        for obj in objects_to_remove:
            print(f"removiong this {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)
    
    #makes sure to remove  all the tree objects from envy/ ufo tree
    for obj in bpy.data.objects:        
        if "_TRUNK" in obj.name and "lpy_envy" in obj.name.lower():
            print(f"first removal did not work, removing for: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)
    
        if "_TRUNK" in obj.name and "lpy_ufo" in obj.name.lower():
            print(f"first removal did not work, removing for: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)

""" removes all the tree object in the scene with a given tree idx and remove tree objects based on id """
def helper_all_tree_objects(tree_idx, remove_envy, remove_ufo):
    root = f"tree{tree_idx}"

    to_delete = [
        obj for obj in bpy.data.objects
        if obj.name.startswith(root) and any(tag in obj.name for tag in ("_TRUNK", "_BRANCH", "_SPUR"))
    ]
    if remove_envy:
        to_delete +=[
            obj for obj in bpy.data.objects
            if ("lpy_envy" in obj.name.lower() and "_TRUNK" in obj.name)
        ]
    if remove_ufo:
        to_delete +=[
            obj for obj in bpy.data.objects
            if ("lpy_ufo" in obj.name.lower() and "_TRUNK" in obj.name)
        ]

    seen = set()
    unique = []
    for obj in to_delete:
        if obj.name not in seen:
            seen.add(obj.name)
            unique.append(obj)
    
    for obj in unique:
        print("removing tree object", obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
            

""" renders in the tree object from the metadata according to the orginal location of rendered tree """
def get_tree_object_for_metadata(tree_id, metadata_path, tree_idx, saved_tree_data=None):
    expected_name = TREE_OBJ.format(tree_idx)
    ply_pth = os.path.join(PLY_DIR, f"{tree_id}.ply")
    
    if not os.path.exists(ply_pth):
        print(f"Ply was not found: {ply_pth}")
        return None
      
    #load in the correct materail and location
    original_location = None
    original_material = None

    if saved_tree_data and tree_idx in saved_tree_data:
        original_location, original_material = saved_tree_data[tree_idx]
        print(f"using the locaation{original_location} and material { original_material}")
    else:
        existing = bpy.data.objects.get(expected_name)
        if existing: 
            original_location = existing.matrix_world.to_translation().copy()
            if getattr(existing.data, "materials", None) and existing.data.materials:
                original_material = existing.data.materials[0]
    
    remove_all_tree_objects()

    #makes sure to delete all tree objects before rendering

    rem_tree = f"{tree_id}_TRUNK"
    rem_obj = bpy.data.objects.get(rem_tree)
    if rem_obj:
        print(f"removing these current tree objects{rem_obj.name}")
        bpy.data.objects.remove(rem_obj, do_unlink=True)

    #imports the ply file
    print(f"import the ply: {os.path.basename(ply_pth)}")
    tree_obj = load_ply_into_blender(ply_pth, expected_name)
    if tree_obj is None:
        print(f"failed to import: {ply_pth}")
        return None
    
    #restores the position
    if original_location is not None:
        tree_obj.location = original_location
    elif saved_tree_data and 0 in saved_tree_data:
        tree_obj.location = saved_tree_data[0][0]
    else:
         tree_obj.location = (0.0, 0.0, 0.0)

    material_add = original_material

    if material_add is None and saved_tree_data:
        for fall_idx in (0, 1):
            if fall_idx in saved_tree_data and saved_tree_data[fall_idx][1]:
                material_add = saved_tree_data[fall_idx][1]
                break
    if material_add:
        tree_obj.data.materials.clear()
        tree_obj.data.materials.append(material_add)
    else:
        print(" No material found the render will not have texture")
    
    tree_obj.hide_viewport = False
    tree_obj.hide_render = False

    bpy.context.view_layer.objects.active = tree_obj
    tree_obj.select_set(True)
    bpy.context.view_layer.update()

    return tree_obj

""" loads in the camera reference from the desired frames ->  """
def load_frame_annotations():
    anns = []
    for pth in REFERENCE_ANNOTATIONS:
        if not os.path.exists(pth):
            print(f"missing annotation: {pth}")
            continue
        with open(pth, "r") as f:
            anns.append(json.load(f))
    return anns


""" Setup combined depth + segmentation masks compositor """
def setup_depth_and_masks(scene, depth_output_dir, mask_output_dir, depth_prefix, mask_prefix):
    # Enable passes
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

    # --- Object index output ---
    idx_sock = rl.outputs.get("IndexOB")
    if idx_sock is None:
        raise RuntimeError("RenderLayers has no 'IndexOB' output. Are you using Cycles and Index pass enabled?")

    # --- Tree mask: ID==1 ---
    tree_mask = nodes.new("CompositorNodeIDMask")
    tree_mask.index = PASS_INDEX_TREE
    tree_mask.location = (-150, 100)
    links.new(idx_sock, tree_mask.inputs["ID value"])

    out_tree = nodes.new("CompositorNodeOutputFile")
    out_tree.base_path = mask_output_dir
    out_tree.format.file_format = "PNG"
    out_tree.format.color_mode = "BW"
    out_tree.file_slots[0].path = f"{mask_prefix}_tree_"
    out_tree.location = (200, 80)
    links.new(tree_mask.outputs["Alpha"], out_tree.inputs[0])

    # --- Union mask: ID==1 OR ID==2 ---
    # Make two ID masks and add them, then clamp.
    bg_mask = nodes.new("CompositorNodeIDMask")
    bg_mask.index = PASS_INDEX_BACKGROUND
    bg_mask.location = (-150, -80)
    links.new(idx_sock, bg_mask.inputs["ID value"])

    add = nodes.new("CompositorNodeMath")
    add.operation = "ADD"
    add.location = (20, 0)
    links.new(tree_mask.outputs["Alpha"], add.inputs[0])
    links.new(bg_mask.outputs["Alpha"], add.inputs[1])

    clamp = nodes.new("CompositorNodeMath")
    clamp.operation = "MINIMUM"
    clamp.inputs[1].default_value = 1.0
    clamp.location = (120, 0)
    links.new(add.outputs[0], clamp.inputs[0])

    out_union = nodes.new("CompositorNodeOutputFile")
    out_union.base_path = mask_output_dir
    out_union.format.file_format = "PNG"
    out_union.format.color_mode = "BW"
    out_union.file_slots[0].path = f"{mask_prefix}_union_"
    out_union.location = (200, -120)
    links.new(clamp.outputs[0], out_union.inputs[0])


""" makes sure to replace and print the lastest depth file"""
def new_depthfile(depth_dir):
    if not os.path.isdir(depth_dir):
        return None
    
    exrs = [
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.lower().endswith(".exr")
    ]

    return max(exrs, key=os.path.getmtime) if exrs else None

"""makes sure directory exists if not creates it """
def ensure_tree_dirs(tree_id):
    rgb_dir = os.path.join(RGB_DIR, tree_id)
    depth_dir = os.path.join(DEPTH_DIR, tree_id)
    ann_dir = os.path.join(ANN_DIR, tree_id)
    mask_dir = os.path.join(MASK_DIR, tree_id)

    os.makedirs(rgb_dir, exist_ok = True)
    os.makedirs(depth_dir, exist_ok = True)
    os.makedirs(ann_dir, exist_ok = True)
    os.makedirs(mask_dir, exist_ok = True)

    return rgb_dir, depth_dir, ann_dir, mask_dir

""" Set pass indices for segmentation masks """
def set_pass_indices(tree_obj, union_objects):
    """
    Set pass indices for segmentation:
    - Tree: pass_index = 1
    - Background objects (posts, wires, ground): pass_index = 2
    """
    # Set tree pass index
    if tree_obj:
        tree_obj.pass_index = PASS_INDEX_TREE
        print(f"  Set pass_index={PASS_INDEX_TREE} for tree: {tree_obj.name}")
    
    # Set background objects pass index
    found_count = 0
    for obj_name in union_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.pass_index = PASS_INDEX_BACKGROUND
            found_count += 1
            print(f"  Set pass_index={PASS_INDEX_BACKGROUND} for {obj_name}")
    
    if found_count == 0:
        print(f"  ⚠ Warning: No background objects found to set pass_index!")
    else:
        print(f"  Set pass_index for {found_count} background objects")

""" Get background object names (posts, wires, ground) """
def get_background_objects():
    """Find all background objects: posts, wires, and ground."""
    background_names = []
    
    # Posts
    for i in range(10):  # Check up to 10 posts
        post_name = f"post{i}"
        if bpy.data.objects.get(post_name):
            background_names.append(post_name)
    
    # Wires (wire0, wire1, wire_01, etc.)
    for obj in bpy.data.objects:
        name_lower = obj.name.lower()
        if "wire" in name_lower and obj.type in {"MESH", "CURVE"}:
            background_names.append(obj.name)
    
    # Ground
    for obj in bpy.data.objects:
        name_lower = obj.name.lower()
        if "ground" in name_lower and obj.type == "MESH":
            background_names.append(obj.name)
    
    print(f"Found {len(background_names)} background objects: {background_names}")
    return background_names

def render_tree(scene, use_cycles_for_masks=False): 
    """
    Set up render settings.
    
    Args:
        use_cycles_for_masks: If True, use Cycles engine (slower but supports Object Index pass).
                              If False, use EEVEE (faster but may not support Object Index).
    """
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    

def render_tree_fixed_shots(scene, cam_obj, tree_id, tree_obj, ref_annotations, K):
    """
    Render exactly 3 RGB + metric-depth images + segmentation masks for one tree using 3 fixed camera poses.

    No jitter. No random pixel sampling. No roll noise. No ray-aiming.
    Just: pose -> render -> save.
    """

    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True

    # --- Make only this tree visible (avoids other trees leaking into depth/rgb) ---
    for obj in bpy.data.objects:
        if obj.name.startswith("tree") and "_TRUNK" in obj.name:
            is_target = (obj == tree_obj)
            obj.hide_viewport = not is_target
            obj.hide_render = not is_target

    tree_rgb_dir, tree_depth_dir, tree_ann_dir, tree_mask_dir = ensure_tree_dirs(tree_id)

    # Get background objects (posts, wires, ground)
    background_objects = get_background_objects()

    # Set pass indices for segmentation
    set_pass_indices(tree_obj, background_objects)


    # Render exactly 3 shots (or however many refs you pass)
    for shot_idx, ref_ann in enumerate(ref_annotations[:3], start=1):
        # --- Set fixed camera pose from the annotation ---
        cam_loc = ref_ann["camera"]["location"]
        cam_rot = ref_ann["camera"]["rotation_euler"]

        cam_obj.location = cam_loc
        cam_obj.rotation_mode = "XYZ"
        cam_obj.rotation_euler = cam_rot

        bpy.context.view_layer.update()

        # --- File names ---
        rgb_name   = f"{tree_id}_shot{shot_idx:02d}.png"
        depth_name = f"{tree_id}_shot{shot_idx:02d}.exr"
        ann_name   = f"{tree_id}_shot{shot_idx:02d}.json"
        mask_prefix = f"{tree_id}_shot{shot_idx:02d}"

        rgb_path   = os.path.join(tree_rgb_dir, rgb_name)
        depth_path = os.path.join(tree_depth_dir, depth_name)
        ann_path   = os.path.join(tree_ann_dir, ann_name)

        # --- Setup combined depth + masks compositor ---
        setup_depth_and_masks(
            scene, 
            tree_depth_dir, 
            tree_mask_dir, 
            f"{tree_id}_shot{shot_idx:02d}",
            mask_prefix
        )
        
        # Use unique frame per shot
        scene.frame_set(shot_idx)

        # --- Render RGB PNG ---
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)

        # --- Render multilayer EXR (this is what gives ViewLayer.Depth.Z) ---
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
        scene.render.filepath = depth_path
        bpy.ops.render.render(write_still=True)

        actual_depth = depth_path

        
        # Update scene to ensure compositor outputs are written
        bpy.context.view_layer.update()


        # --- Find mask files (they may have frame numbers appended) ---
        # Look for mask files with the prefix
        mask_tree_path = None
        mask_union_path = None
        mask_visible_path = None
        
        if os.path.exists(tree_mask_dir):
            for fname in os.listdir(tree_mask_dir):
                if fname.startswith(f"{mask_prefix}_tree_") and fname.endswith(".png"):
                    mask_tree_path = os.path.join(tree_mask_dir, fname)
                elif fname.startswith(f"{mask_prefix}_union_") and fname.endswith(".png"):
                    mask_union_path = os.path.join(tree_mask_dir, fname)
                elif fname.startswith(f"{mask_prefix}_visible_tree_") and fname.endswith(".png"):
                    mask_visible_path = os.path.join(tree_mask_dir, fname)
        
        # Fallback to expected names if not found
        if mask_tree_path is None:
            mask_tree_path = os.path.join(tree_mask_dir, f"{mask_prefix}_tree_0001.png")
        if mask_union_path is None:
            mask_union_path = os.path.join(tree_mask_dir, f"{mask_prefix}_union_0001.png")
        if mask_visible_path is None:
            mask_visible_path = os.path.join(tree_mask_dir, f"{mask_prefix}_visible_tree_0001.png")

        # --- Minimal annotation (optional) ---
        loc = cam_obj.matrix_world.to_translation()
        eul = cam_obj.matrix_world.to_euler("XYZ")

        ann = {
            "tree_id": tree_id,
            "shot": shot_idx,
            "rgb_path": rgb_path,
            "depth_path": actual_depth,
            "masks": {
                "tree_only": mask_tree_path if os.path.exists(mask_tree_path) else None,
                "union": mask_union_path if os.path.exists(mask_union_path) else None,
                "visible_tree": mask_visible_path if os.path.exists(mask_visible_path) else None,
            },
            "camera": {
                "location": [loc.x, loc.y, loc.z],
                "rotation_euler": [eul.x, eul.y, eul.z],
                "intrinsics": {"width": W, "height": H, "K": K},
            },
             "reference": {
                    "post0": POST0_NAME,
                    "post1": POST1_NAME,
             },
            "tree_object": tree_obj.name,
            "background_objects": background_objects,
        }

        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        # Check which masks were actually created
        masks_created = []
        if os.path.exists(mask_tree_path):
            masks_created.append("tree")
        if os.path.exists(mask_union_path):
            masks_created.append("union")
        if os.path.exists(mask_visible_path):
            masks_created.append("visible")
        
        mask_info = ""
        if masks_created:
            mask_info = f", masks: {', '.join(masks_created)}"
        else:
            mask_info = ", ⚠ NO MASKS CREATED - check pass indices and Object Index pass"
            # Debug: list what files are in the mask directory
            if os.path.exists(tree_mask_dir):
                mask_files = [f for f in os.listdir(tree_mask_dir) if f.endswith('.png')]
                if mask_files:
                    print(f"    Found {len(mask_files)} PNG files in mask dir: {mask_files[:5]}")
                else:
                    print(f"    No PNG files found in mask directory: {tree_mask_dir}")
        
        print(f"✓ Rendered shot {shot_idx}/3 for {tree_id}: {rgb_name}, {depth_name}{mask_info}")
        
        # Optional: visualize depth (uncomment if you have a visualization function)
        # if os.path.exists(depth_path):
        #     visualize_depth_minimal(depth_path)


def main():
    # 1) Create output folders (one time)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(ANN_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # 2) Get Blender scene + camera
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    cam_obj = bpy.data.objects.get(CAM)
    cam_obj.data.lens = 28.0   # smaller = wider (e.g., 18 ultra-wide, 35 normal)
    if cam_obj is None:
        raise RuntimeError(f"Camera '{CAM}' not found in the .blend")

    scene.camera = cam_obj

    # 3) Render settings (resolution, png format, engine, etc.)
    # Set use_cycles_for_masks=True if EEVEE doesn't support Object Index pass
    render_tree(scene, use_cycles_for_masks=True)

    # 4) Load the 3 reference camera annotations (frames 0009/0016/0023)
    ref_annotations = load_frame_annotations()
    if len(ref_annotations) < 3:
        raise RuntimeError("Need 3 reference frames. Missing annotation JSONs?")

    # 5) Compute intrinsics matrix K ONCE (same for all shots)
    # If you already have a function for this, call it here.
    # Otherwise pass a placeholder or build K manually.
    # Example: K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    # --- Scale intrinsics from reference JSON ---
    ref_intr = ref_annotations[0]["camera"]["intrinsics"]

    W0 = ref_intr["width"]
    H0 = ref_intr["height"]
    K0 = ref_intr["K"]

    sx = W / W0
    sy = H / H0

    K = [
        [K0[0][0] * sx, 0.0,           K0[0][2] * sx],
        [0.0,           K0[1][1] * sy, K0[1][2] * sy],
        [0.0,           0.0,           1.0],
    ]

    print("Scaled intrinsics K:")
    for row in K:
        print(row)

    # 6) Find trees to process (tree_id, metadata_path, file_num)
    metadata_files = find_all_metadata_files()
    if not metadata_files:
        print("No metadata found. Nothing to render.")
        return

    # Optional: filter to only specific trees (like envy)
    # target = {"lpy_envy_00000", "lpy_envy_00001"}
    # metadata_files = [t for t in metadata_files if t[0] in target]

    # 7) Save original locations/materials from the .blend BEFORE deleting anything
    # This assumes you already have placeholder objects tree0_TRUNK, tree1_TRUNK, etc.
    saved_tree_data = {}
    for idx in range(len(metadata_files)):
        obj_name = TREE_OBJ.format(idx)
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue

        loc = obj.matrix_world.to_translation().copy()
        mat = obj.data.materials[0] if obj.data.materials else None
        saved_tree_data[idx] = (loc, mat)

    # 8) Clean old tree objects out of the scene once
    remove_all_tree_objects()

    # 9) Loop over each tree: import PLY -> render 3 fixed shots
    for tree_idx, (tree_id, metadata_path, file_num) in enumerate(metadata_files):
        print(f"\n[{tree_idx+1}/{len(metadata_files)}] Rendering tree: {tree_id}")

        # Import + name it tree{idx}_TRUNK, restore location/material
        tree_obj = get_tree_object_for_metadata(
            tree_id=tree_id,
            metadata_path=metadata_path,
            tree_idx=tree_idx,
            saved_tree_data=saved_tree_data
        )

        if tree_obj is None:
            print(f"⚠ Skipping {tree_id} (failed import)")
            continue

        # Render exactly 3 images (and 3 metric depth EXRs)
        render_tree_fixed_shots(
            scene=scene,
            cam_obj=cam_obj,
            tree_id=tree_id,
            tree_obj=tree_obj,
            ref_annotations=ref_annotations,
            K=K
        )

        # Optional: remove tree after rendering to keep memory low
        # bpy.data.objects.remove(tree_obj, do_unlink=True)

    print("\n✓ Done rendering all trees.")


if __name__ == "__main__":
    main()
