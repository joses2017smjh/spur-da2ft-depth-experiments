"""Stub: source was deleted; loads classes from precompiled bytecode."""
import importlib.util, sys

_PYC = "/nfs/hpc/share/sanchej7/Computer_Vision/MVP_MODEL/_mvp_precompiled.pyc"

_spec = importlib.util.spec_from_file_location("_mvp_precompiled", _PYC)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["_mvp_precompiled"] = _mod
_spec.loader.exec_module(_mod)

ConvBlock             = _mod.ConvBlock
Down                  = _mod.Down
Up                    = _mod.Up
UNetEncoder           = _mod.UNetEncoder
UNetDecoder           = _mod.UNetDecoder
PoseProject           = _mod.PoseProject
MVUNetPoseConcat      = _mod.MVUNetPoseConcat
masked_smoothl1_3view = _mod.masked_smoothl1_3view
masked_rmse_3view     = _mod.masked_rmse_3view
masked_silog_3view    = _mod.masked_silog_3view

# Reconstructed from annotation sampling (original source unrecoverable).
# Mean camera height ≈ 1.43 m; horizontal radius ≈ 11.66 m; uniform orbit → 0.0.
_CAM_Z_BASE     = 1.43
_CAM_ROT_Z_BASE = 0.0
_CAM_R_BASE     = 11.66
