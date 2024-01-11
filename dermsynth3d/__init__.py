# Authors: Ashish Sinha, Jeremy Kawahara
# Organization: Medical Image Analysis Lab, Computing Science, Simon Fraser University
# Date created: Jan 2024

from .version import __version__
import sys

sys.path.append("..")
sys.path.append("../skin3d/")
sys.path.append("../skin3d/skin3d/")

from skin3d import *
from .tools.select_location import SelectAndPaste
from .tools.generate2d import Generate2DViews, Generate2DHelper
from .tools.synthesize import Synthesize2D, min_scale_size
from .tools.renderer import (
    MeshRendererPyTorch3D,
    camera_pos_from_normal,
    camera_world_position,
)
from .tools.blend_lesions import BlendLesions
from .datasets import *
from .deepblend import Blended3d, DeepImageBlend, DeepTextureBlend3d, Blend
from .models import faster_rcnn_texture_model, inference_multitask
from .losses import (
    compute_results_segmentation,
    compute_results,
    dice,
    dice_loss,
    dice_score,
)

__all__ = [
    "ImageDataset",
    "SynthDataset_Detection",
    "RealDataset_Detection",
    "PratheepanSkinDataset",
    "BinarySegementationDataset",
    "HGRDataset",
    "SynthDataset",
    "FitzDataset",
    "Background2d",
    "NoGTDataset",
    "Ph2Dataset",
    "Fitz17KAnnotations",
    "DermoFit",
    "SynthesizeDataset",
    "Blended3d",
    "BlendLesions",
    "Generate2DViews",
    "Generate2DHelper",
    "SelectAndPaste",
    "Synthesize2D",
    "MeshRendererPyTorch3D",
    "camera_pos_from_normal",
    "camera_world_position",
    "min_scale_size",
]
