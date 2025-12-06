"""
Simplified rendering module that loads assets directly from processed_textures/
and renders RGB, depth, anatomy, and segmentation maps.
"""

import os
import sys
import numpy as np
import torch
import trimesh
from PIL import Image
from PIL import Image
from uuid import uuid4
# Add paths for imports
PROJECT_ROOT_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT_ABS)
sys.path.insert(0, os.path.join(PROJECT_ROOT_ABS, 'skin3d'))

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)

from dermsynth3d.tools.renderer import MeshRendererPyTorch3D
from dermsynth3d.utils.image import load_image
from dermsynth3d.utils.tensor import pil_to_tensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Utility functions for converting outputs (copied from main.py)
COLOR_LABELS = {
    0: (0., 0., 0.), 1: (174., 199., 232.), 2: (152., 223., 138.),
    3: (31., 119., 180.), 4: (255., 187., 120.), 5: (188., 189., 34.),
    6: (140., 86., 75.), 7: (255., 152., 150.),
}


class SimpleRenderer:
    """Simplified renderer that loads assets directly from processed_textures/"""
    
    def __init__(
        self,
        mesh_filename: str,
        texture_path: str,
        anatomy_labels_path: str,
        device=DEVICE,
        lesion_texture_mask_path: str = None,
        nonskin_texture_mask_path: str = None,
    ):
        self.device = device
        self.mesh_filename = mesh_filename
        
        # Load mesh
        self.mesh = load_objs_as_meshes([mesh_filename], device=self.device)
        
        # Load texture
        texture_img = load_image(texture_path)
        texture_tensor = pil_to_tensor(texture_img).to(self.device)
        self.texture_shape = texture_tensor.shape[:2]  # (H, W)
        
        # Set texture on mesh
        self.mesh.textures = TexturesUV(
            maps=texture_tensor.unsqueeze(0),
            verts_uvs=self.mesh.textures.verts_uvs_padded(),
            faces_uvs=self.mesh.textures.faces_uvs_padded(),
        )
        
        # Store original texture for later use
        self.texture_tensor = texture_tensor
        
        # Load anatomy labels
        self.vertices_to_anatomy = torch.from_numpy(
            np.load(anatomy_labels_path)
        ).squeeze().to(self.device)
        # breakpoint()
        # Load lesion and nonskin masks if available, otherwise create empty ones
        if lesion_texture_mask_path and os.path.exists(lesion_texture_mask_path):
            lesion_img = load_image(lesion_texture_mask_path, mode="L")
            self.texture_lesion_mask = pil_to_tensor(lesion_img).to(self.device)
            if self.texture_lesion_mask.ndim > 2:
                self.texture_lesion_mask = self.texture_lesion_mask.squeeze()
        else:
            # Create empty lesion mask
            self.texture_lesion_mask = torch.zeros(
                self.texture_shape, dtype=torch.float32, device=self.device
            )
        
        if nonskin_texture_mask_path and os.path.exists(nonskin_texture_mask_path):
            nonskin_img = load_image(nonskin_texture_mask_path, mode="L")
            self.nonskin_texture_mask_tensor = pil_to_tensor(nonskin_img).to(self.device)
            if self.nonskin_texture_mask_tensor.ndim > 2:
                self.nonskin_texture_mask_tensor = self.nonskin_texture_mask_tensor.squeeze()
        else:
            # Create empty nonskin mask (all skin)
            self.nonskin_texture_mask_tensor = torch.zeros(
                self.texture_shape, dtype=torch.float32, device=self.device
            )
        
        # Initialize renderer
        self.mesh_renderer = MeshRendererPyTorch3D(
            self.mesh,
            self.device,
            config=None,
        )
        
        # Store original renderer settings
        self.original_texture_set = False
        self.current_light_location = None  # Store current light location for mask rendering
        
        # Extract subject_id from mesh filename (e.g., "006-f-run" from path)
        mesh_dir = os.path.dirname(mesh_filename)
        self.subject_id = os.path.basename(mesh_dir)
    
    def set_camera_and_render(
        self,
        dist=1.0,
        elev=0.0,
        azim=0.0,
        fov=30.0,
        view_size=(224, 224),
        light_location=[[0.0, 0.0, -3.0]],
        ambient_color=[[0.5, 0.5, 0.5]],
        diffuse_color=[[0.5, 0.5, 0.5]],
        specular_color=[[0.0, 0.0, 0.0]],
        shininess=[64],
    ):
        """Set camera parameters and render the view"""
        
        # Set view parameters
        self.mesh_renderer.precompute_view_parameters(
            view_size=view_size,
            at=(0.0, 0.0, 0.0),
            dist=dist,
            elev=elev,
            azim=azim,
            znear=0.01,
        )
        
        # Set lights - convert list to tuple if needed
        if isinstance(light_location, list) and len(light_location) > 0:
            if isinstance(light_location[0], list):
                light_location = tuple(light_location[0])
            else:
                light_location = tuple(light_location)
        
        # Extract single values from lists if needed
        ambient = ambient_color[0] if isinstance(ambient_color[0], list) else ambient_color
        diffuse = diffuse_color[0] if isinstance(diffuse_color[0], list) else diffuse_color
        specular = specular_color[0] if isinstance(specular_color[0], list) else specular_color
        shininess_val = shininess[0] if isinstance(shininess, list) else shininess
        
        # Convert to tuples
        ambient = tuple(ambient) if not isinstance(ambient, tuple) else ambient
        diffuse = tuple(diffuse) if not isinstance(diffuse, tuple) else diffuse
        specular = tuple(specular) if not isinstance(specular, tuple) else specular
        
        # Store light location for later use in mask rendering
        self.current_light_location = light_location
        
        # Set lights using precompute_light_parameters
        self.mesh_renderer.precompute_light_parameters(
            light_location=light_location,
            ambient_color=ambient,
            diffuse_color=diffuse,
            specular_color=specular,
        )
        
        # Set materials using precompute_material_parameters
        self.mesh_renderer.precompute_material_parameters(
            ambient_color=ambient,
            specular_color=specular,
            diffuse_color=diffuse,
            shininess=shininess_val,
        )
        
        # Initialize renderer
        self.mesh_renderer.initialize_renderer()
        
        # Compute fragments for depth/anatomy
        self.mesh_renderer.compute_fragments()
    
    def render_rgb(self):
        """Render RGB image"""
        rgb = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        return rgb
    
    def render_depth(self):
        """Render depth map"""
        depth = self.mesh_renderer.depth_view(asnumpy=True)
        # Handle background (negative values or zeros)
        depth = np.where(depth > 0, depth, 0)
        return depth
    
    def render_anatomy(self):
        """Render anatomy map"""
        anatomy = self.mesh_renderer.anatomy_image(self.vertices_to_anatomy)
        return anatomy
    
    def _set_mask_parameters(self):
        """Set renderer parameters for mask rendering"""
        # Use stored light location, or camera position, or default
        if self.current_light_location is not None:
            light_location = tuple(self.current_light_location) if isinstance(self.current_light_location, (list, np.ndarray)) else self.current_light_location
        elif self.mesh_renderer.camera_pos is not None:
            light_location = tuple(self.mesh_renderer.camera_pos)
        else:
            # Default light location if nothing is set
            light_location = (0.0, 0.0, -3.0)
        
        self.mesh_renderer.precompute_light_parameters(
            ambient_color=(1, 1, 1),
            specular_color=(0, 0, 0),
            diffuse_color=(0, 0, 0),
            light_location=light_location,
        )
        self.mesh_renderer.precompute_material_parameters(
            ambient_color=(1, 1, 1),
            specular_color=(0, 0, 0),
            diffuse_color=(0, 0, 0),
            shininess=0,
        )
        self.mesh_renderer.initialize_renderer()
    
    def _set_image_parameters(self, light_location, ambient, specular, diffuse, shininess):
        """Restore renderer parameters for image rendering"""
        self.mesh_renderer.precompute_light_parameters(
            ambient_color=ambient,
            specular_color=specular,
            diffuse_color=diffuse,
            light_location=light_location,
        )
        self.mesh_renderer.precompute_material_parameters(
            ambient_color=ambient,
            specular_color=specular,
            diffuse_color=diffuse,
            shininess=shininess,
        )
        self.mesh_renderer.initialize_renderer()
        # Restore original texture
        self.mesh.textures = TexturesUV(
            maps=self.texture_tensor.unsqueeze(0),
            verts_uvs=self.mesh.textures.verts_uvs_padded(),
            faces_uvs=self.mesh.textures.faces_uvs_padded(),
        )
    
    def render_skin_mask(self):
        """Render skin mask (inverse of nonskin mask)"""
        # Set nonskin texture mask - expand to 3 channels
        nonskin_3channel = self.nonskin_texture_mask_tensor.unsqueeze(0)
        if nonskin_3channel.ndim == 3:
            # Add channel dimension and expand to RGB
            nonskin_3channel = nonskin_3channel.unsqueeze(-1)
        if nonskin_3channel.shape[-1] == 1:
            nonskin_3channel = nonskin_3channel.expand(-1, -1, -1, 3)
        
        self.mesh.textures = TexturesUV(
            maps=nonskin_3channel,
            verts_uvs=self.mesh.textures.verts_uvs_padded(),
            faces_uvs=self.mesh.textures.faces_uvs_padded(),
        )
        self._set_mask_parameters()
        
        # Render nonskin mask
        nonskin_mask_2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        # Get body mask (pixels that hit the mesh)
        body_mask = self.mesh_renderer.body_mask().astype(np.float32)
        # Nonskin is where mask > 0.5, skin is the inverse within body
        nonskin_mask = (nonskin_mask_2d[:, :, 0] > 0.5).astype(np.float32)
        skin_mask = body_mask * (1.0 - nonskin_mask)  # Skin = body AND not nonskin
        
        return skin_mask
    
    def render_lesion_mask(self):
        """Render lesion mask"""
        # Set lesion texture mask - expand to 3 channels
        lesion_3channel = self.texture_lesion_mask.unsqueeze(0)
        if lesion_3channel.ndim == 3:
            # Add channel dimension and expand to RGB
            lesion_3channel = lesion_3channel.unsqueeze(-1)
        if lesion_3channel.shape[-1] == 1:
            lesion_3channel = lesion_3channel.expand(-1, -1, -1, 3)
        
        self.mesh.textures = TexturesUV(
            maps=lesion_3channel,
            verts_uvs=self.mesh.textures.verts_uvs_padded(),
            faces_uvs=self.mesh.textures.faces_uvs_padded(),
        )
        self._set_mask_parameters()
        
        # Render lesion mask
        lesion_mask_2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        # Get body mask
        body_mask = self.mesh_renderer.body_mask().astype(np.float32)
        # Lesion is where mask > 0.5 within body
        lesion_mask = (lesion_mask_2d[:, :, 0] > 0.5).astype(np.float32)
        lesion_mask = body_mask * lesion_mask  # Lesion must be within body
        
        return lesion_mask
    
    def render_segmentation(self, light_location, ambient, specular, diffuse, shininess):
        """Render multi-label segmentation map: lesion (0), skin (1), nonskin (2)"""
        # Render skin and lesion masks
        skin_mask = self.render_skin_mask()
        lesion_mask = self.render_lesion_mask()
        
        # Restore image rendering parameters
        self._set_image_parameters(light_location, ambient, specular, diffuse, shininess)
        
        # Get body mask for background
        body_mask = self.mesh_renderer.body_mask()
        
        # Create multi-label segmentation map following make_masks logic
        # Channel 0: Lesion
        # Channel 1: Skin (without lesion) - using skin_mask_no_lesion logic
        # Channel 2: Nonskin (background or non-skin parts of body)
        seg_map = np.zeros((skin_mask.shape[0], skin_mask.shape[1], 3), dtype=np.float32)
        seg_map[:, :, 0] = lesion_mask * COLOR_LABELS[1][0]  # Lesion channel
        # Image.fromarray((seg_map * 255).astype(np.uint8)).save(f"Lesion_seg_map_{uuid4()}.png")
        
        # Skin without lesion: skin AND not lesion
        skin_no_lesion = skin_mask * (1.0 - lesion_mask)
        seg_map[:, :, 1] = skin_no_lesion * COLOR_LABELS[2][0]  # Skin channel
        # Image.fromarray((seg_map * 255).astype(np.uint8)).save(f"Skin_seg_map_{uuid4()}.png")

        seg_map[:, :, 2] = (1.0 - skin_mask).astype(np.float32) * COLOR_LABELS[0][0]  # Nonskin channel 
        # Image.fromarray((seg_map * 255).astype(np.uint8)).save(f"Nonskin_seg_map_{uuid4()}.png")
        
        return seg_map
    
    def render_all(self, light_location, ambient, specular, diffuse, shininess):
        """Render all maps: RGB, depth, anatomy, segmentation"""
        rgb = self.render_rgb()
        depth = self.render_depth()
        anatomy = self.render_anatomy()
        segmentation = self.render_segmentation(light_location, ambient, specular, diffuse, shininess)
        
        return rgb, depth, anatomy, segmentation


def get_texture_path(
    processed_textures_dir: str,
    subject_id: str,
    texture_type: str,
    num_lesions: int,
    mesh_dir: str,
):
    """Get the path to the texture file based on type and number of lesions"""
    if texture_type == "No Lesion" or num_lesions == 0:
        # Use original texture from mesh directory
        return os.path.join(mesh_dir, "model_highres_0_normalized.png")
    else:
        lesion_type_map = {
            "Pasted Lesion": "pasted",
            "Blended Lesion": "blended",
            "Dilated Lesion": "dilated"
        }
        lesion_str = lesion_type_map.get(texture_type, "blended")
        return os.path.join(
            processed_textures_dir,
            subject_id,
            f"model_highres_0_normalized_{lesion_str}_lesion_{num_lesions}.png"
        )


def get_anatomy_labels_path(anatomy_labels_dir: str, subject_id: str):
    """Get the path to anatomy labels file"""
    return os.path.join(anatomy_labels_dir, subject_id, "vertslabels_scan.npy")

def to_simple_anatomy(anatomy):
    """Convert anatomy labels to simplified labels"""
    for i in range(16+1):
        if i in [0,1]:
            continue
        if i in [2,3]:
            anatomy[anatomy==i] = 2
        if i == 4:
            anatomy[anatomy==i] = 3
        if i in [5,6,7,8]:
            anatomy[anatomy==i] = 4
        if i in [9,10]:
            anatomy[anatomy==i] = 5
        if i in [11,12,13,14]:
            anatomy[anatomy==i] = 6
        if i in [15,16]:
            anatomy[anatomy==i] = 7
    return anatomy


def convert_anatomy_to_rgb(anatomy):
    """Convert anatomy map to RGB visualization"""
    # Ensure anatomy is 2D (H, W)
    if anatomy.ndim == 3 and anatomy.shape[-1] == 1:
        anatomy = anatomy.squeeze(axis=-1)
    elif anatomy.ndim != 2:
        raise ValueError(f"Expected anatomy to be 2D or 3D (H,W,1), but got shape {anatomy.shape}")

    anatomy = to_simple_anatomy(anatomy)
    anatomy_rgb = np.zeros((anatomy.shape[0], anatomy.shape[1], 3), dtype=np.uint8)
    for k, v in COLOR_LABELS.items():
        anatomy_rgb[anatomy == k] = np.array(v, dtype=np.uint8)
    return anatomy_rgb


def convert_depth_to_rgb(depth):
    """Convert depth map to RGB visualization"""
    if np.all(depth == 0):
        return np.full((depth.shape[0], depth.shape[1], 3), 255, dtype=np.uint8)
    
    # Ensure depth is 2D (H, W)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(axis=-1)
    elif depth.ndim != 2:
        raise ValueError(f"Expected depth to be 2D or 3D (H,W,1), but got shape {depth.shape}")

    mask = depth != 0  # mask is (H, W)
    disp_map = np.zeros_like(depth, dtype=float)  # disp_map is (H, W)
    
    # Avoid division by zero if depth[mask] is empty
    if mask.sum() > 0:
        disp_map[mask] = 1 / depth[mask]

    vmax = np.percentile(disp_map[mask], 95) if mask.sum() > 0 else 1.0
    vmin = np.percentile(disp_map[mask], 5) if mask.sum() > 0 else 0.0
    
    if vmax == vmin:
        normalized_map = np.zeros_like(disp_map)
    else:
        normalized_map = (disp_map - vmin) / (vmax - vmin)
    
    import matplotlib.cm as cm
    magma_cmap = cm.get_cmap('magma')
    colormapped_im = (magma_cmap(normalized_map)[:, :, :3] * 255).astype(np.uint8)
    
    # Create a 3-channel boolean mask from the 2D mask
    three_channel_mask = np.stack([~mask, ~mask, ~mask], axis=-1)
    colormapped_im[three_channel_mask] = 255  # Assign white to masked out areas
    return colormapped_im

if __name__ == "__main__":
    # breakpoint()
    renderer = SimpleRenderer(
        mesh_filename="../data/3dbodytex-1.1-highres/006-f-run/model_highres_0_normalized.obj",
        texture_path="../data/3dbodytex-1.1-highres/006-f-run/model_highres_0_normalized.png",
        anatomy_labels_path="../data/bodytex_anatomy_labels/006-f-run/vertslabels_scan.npy",
        device=DEVICE,
    )
    renderer.set_camera_and_render()
    renderer.render_all(
        light_location=[0.0, 0.0, -3.0],
        ambient=[0.5, 0.5, 0.5],
        specular=[0.0, 0.0, 0.0],
        diffuse=[0.5, 0.5, 0.5],
        shininess=[64],
    )
    renderer.render_skin_mask()
    renderer.render_lesion_mask()
    renderer.render_segmentation()
    renderer.render_all()