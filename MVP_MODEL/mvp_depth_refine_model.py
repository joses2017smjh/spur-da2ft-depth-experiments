"""Stub: source was deleted; loads classes from precompiled bytecode."""
import importlib.util, sys

_PYC = "/nfs/hpc/share/sanchej7/Computer_Vision/MVP_MODEL/_mvp_precompiled.pyc"

# Reuse the already-loaded module if mvp_depth_model_Unet.py loaded it first.
if "_mvp_precompiled" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("_mvp_precompiled", _PYC)
    _mod  = importlib.util.module_from_spec(_spec)
    sys.modules["_mvp_precompiled"] = _mod
    _spec.loader.exec_module(_mod)

_mod = sys.modules["_mvp_precompiled"]

ConvBlock              = _mod.ConvBlock
Down                   = _mod.Down
Up                     = _mod.Up
UNetEncoder            = _mod.UNetEncoder
UNetDecoder            = _mod.UNetDecoder
PoseProject            = _mod.PoseProject
MVUNetPoseConcat       = _mod.MVUNetPoseConcat
DepthSideBranch        = _mod.DepthSideBranch
DINOv2ViTLEncoder      = _mod.DINOv2ViTLEncoder
DINODecoder            = _mod.DINODecoder
MVDINOv2PoseConcat     = _mod.MVDINOv2PoseConcat
masked_smoothl1_3view  = _mod.masked_smoothl1_3view
masked_rmse_3view      = _mod.masked_rmse_3view
masked_silog_3view     = _mod.masked_silog_3view
gt_warp_consistency_loss = _mod.gt_warp_consistency_loss
