from functools import partial
import gradio as gr
import pdb
from PIL import Image
import numpy as np
import gradio as gr
import torch
import os
import fire
import multiprocessing as mp
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "DermSynth3D"))
sys.path.append(os.path.join(os.path.dirname(__file__), "DermSynth3D", "dermsynth3d"))
sys.path.append(os.path.join(os.path.dirname(__file__), "DermSynth3D", "skin3d"))

import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import math
from trimesh import transformations as tf
import os
from math import pi
import matplotlib.pyplot as plt
import plotly

import plotly.graph_objects as go
from skimage import io

view_width = 400
view_height = 400

import mediapy as mpy

try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.structures import Meshes

    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        DirectionalLights,
        Materials,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesUV,
        TexturesVertex,
    )

    print("Pytorch3d compiled properly")
except:
    print("Pytorch3d not compiled properly. Install pytorch3d with torch/cuda support")

try:
    sys.path.append("./DermSynth3D/")
    sys.path.append("./DermSynth3D/dermsynth3d/")
    sys.path.append("./DermSynth3D/skin3d/")
    from dermsynth3d import BlendLesions, Generate2DViews, SelectAndPaste
    from dermsynth3d.tools.generate2d import Generate2DHelper
    from dermsynth3d.utils.utils import yaml_loader
    from dermsynth3d.utils.utils import random_bound, make_masks
    from dermsynth3d.tools.synthesize import Synthesize2D
    from dermsynth3d.datasets.synth_dataset import SynthesizeDataset
    from dermsynth3d.tools.renderer import (
        MeshRendererPyTorch3D,
        camera_pos_from_normal,
    )
    from dermsynth3d.deepblend.blend3d import Blended3d
    from dermsynth3d.utils.channels import Target
    from dermsynth3d.utils.tensor import (
        pil_to_tensor,
    )
    from dermsynth3d.utils.colorconstancy import shade_of_gray_cc
    from dermsynth3d.datasets.datasets import Fitz17KAnnotations, Background2d
    from skin3d.skin3d.bodytex import BodyTexDataset

    print("DermSynth3D compiled properly")
except Exception as e:
    print(e)
    print("DermSynth3D not in the path. Make sure to add it to the path.")

_TITLE = """DermSynth3D: A Framework for generating Synthetic Dermatological Images"""
_DESCRIPTION = """
**Step 1**. Select the Mesh, texture map and number of lesions from the dropdown or select an example.</br>
**Step 2**. Selct the number of views to render. </br>
**Step 3** (optional). Randomize the view parameters by clicking on the checkbox.</br>
**Step 4**. Click on the Render Views button to render the views. </br>
"""


deployed = True

if deployed:
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Running on CPU")

global mesh_paths, mesh_names, all_textures, dir_blended_textures, dir_anatomy
global get_no_lesion_path, get_mesh_path, get_mask_path, get_dilated_lesion_path
global get_blended_lesion_path, get_pasted_lesion_path, get_texture_module
global dir_blended_textures, dir_anatomy, dir_background

# File path of the bodytex CSV.
bodytex_csv = "./DermSynth3D/skin3d/data/3dbodytex-1.1-highres/bodytex.csv"
bodytex_df = pd.read_csv(bodytex_csv, converters={"scan_id": lambda x: str(x)})
bodytex = BodyTexDataset(
    df=bodytex_df,
    dir_textures="./DermSynth3D/data/3dbodytex-1.1-highres/",
    dir_annotate="./DermSynth3D/skin3d/data/3dbodytex-1.1-highres/annotations/",
)
# True to use the blended lesions, False to use the pasted lesions.
is_blend = True
background_ds = Background2d(
    dir_images="./DermSynth3D/data/background/IndoorScene/",
    image_filenames=None,
)

from dermsynth3d.utils.anatomy import SimpleAnatomy
color_labels = {
            0: (0., 0., 0.),  # background
    1: (174., 199., 232.), # head
    2: (152., 223., 138.), # torso
    3: (31., 119., 180.), # hips
    4: (255., 187., 120.), # legs
    5: (188., 189., 34.), # feet
    6: (140., 86., 75.), # arms
    7: (255., 152., 150.), # hands
    }


def to_simple_anatomy(anatomy):
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
    anatomy = to_simple_anatomy(anatomy)
    anatomy_rgb = np.zeros((anatomy.shape[0], anatomy.shape[1], 3))
    for k, v in color_labels.items():
        anatomy_rgb[anatomy == k] = v
    return anatomy_rgb.astype(np.uint8)

import PIL.Image as pil
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
def convert_depth_to_rgb(depth):
    mask = depth != 0
    disp_map = 1 / depth
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

def prepare_ds_renderer(
    randomize,
    mesh_name,
    texture_name,
    num_lesion,
    num_views,
    dist,
    elev,
    azim,
    light_pos,
    light_ac,
    light_dc,
    light_sc,
    mat_sh,
    mat_sc,
    device=DEVICE,
):
    mesh_filename = get_mesh_path(mesh_name)
    mesh = load_mesh_and_texture(mesh_name, texture_name, num_lesion, device)
    gr.Info("Preparing for Rendering...")
    mesh_renderer = MeshRendererPyTorch3D(mesh, DEVICE, config=None)
    extension = f"lesion_{num_lesion}"
    # if texture_name != "No Lesion":
    #     extension = f"{texture_name.lower().split(' ')[0]}_lesion_{num_lesion}"
    nevi_exists = os.path.exists(bodytex.annotation_filepath(mesh_name.split("_")[0]))
    gen2d = Generate2DHelper(
        mesh_filename=mesh_filename,
        dir_blended_textures="./hf_demo/lesions/",
        dir_anatomy="./DermSynth3D/data/bodytex_anatomy_labels/",
        fitz_ds=None,  # fitz_ds,
        background_ds=background_ds,
        device=device,
        debug=True,
        bodytex=bodytex,
        blended_file_ext=extension,  # if num_lesion > 0 else "demo",
        config=None,
        is_blended=is_blend,
    )
    # blended3d = Blended3d(
    #     mesh_filename=os.path.join(
    #         "./DermSynth3D/data/3dbodytex-1.1-highres/",
    #         mesh_name,
    #         "model_highres_0_normalized.obj",
    #     ),
    #     device=DEVICE,
    #     dir_blended_textures=dir_blended_textures,
    #     dir_anatomy=dir_anatomy,
    #     extension=extension ,
    # )
    # normal_texture = load_texture_map(
    #     mesh, mesh_name, "No Lesion", 0, device
    # ).maps_padded()
    # if num_lesion > 0:
    #     blended_texture_image = load_texture_map(
    #         mesh, mesh_name, "Blended Lesion", num_lesion, device
    #     ).maps_padded()
    #     pasted_texture_image = load_texture_map(
    #         mesh, mesh_name, "Pasted Lesion", num_lesion, device
    #     ).maps_padded()
    #     dilated_texture_image = load_texture_map(
    #         mesh, mesh_name, "Dilated Lesion", num_lesion, device
    #     ).maps_padded()

    # texture_lesion_mask = blended3d.lesion_texture_mask(astensor=True).to(device)
    # non_skin_texture_mask = blended3d.nonskin_texture_mask(astensor=True).to(device)
    # vertices_to_anatomy = blended3d.vertices_to_anatomy()
    # mesh_renderer.raster_settings = raster_settings
    renderer, cameras, lights, materials = set_rendering_params(
        randomize,
        1,  # num_views,
        dist,
        elev,
        azim,
        light_pos,
        light_ac,
        light_dc,
        light_sc,
        mat_sh,
        mat_sc,
    )
    gr.Info("Successfully prepared renderer.")
    gr.Info("Rendering Images...")
    gr.Info(f"Rendering {num_views} views on {DEVICE}. Please wait...")
    img_count = 0
    view2d = []
    depth2d = []
    anatomy2d = []
    seg2d = []
    view_size = (224, 224)
    gen2d.view_size = view_size
    while img_count < num_views:
        if randomize:
            gr.Info("Finding suitable parameters...")
            success = gen2d.randomize_parameters(config=None, view_size=view_size)
            if not success:
                gr.Info("Could not find suitable parameters. Trying again.")
                continue
        else:
            raster_settings = RasterizationSettings(
                image_size=view_size[0],
                blur_radius=0.0,
                faces_per_pixel=10,
                perspective_correct=True,
            )
            gen2d.mesh_renderer.cameras = cameras
            gen2d.mesh_renderer.lights = lights
            gen2d.mesh_renderer.materials = materials
            gen2d.mesh_renderer.raster_settings = raster_settings
            gen2d.mesh_renderer.initialize_renderer()
            gr.Info("Rasterization in progress...")
            gen2d.mesh_renderer.compute_fragments()
            gr.Info("Successfully rasterized.")
        paste_img, target = gen2d.render_image_and_target(paste_lesion=False)
        if paste_img is None:
            gr.Info(
                "***Not enough skin or unable to paste lesion. Skipping and Retrying."
            )
            print("***Not enough skin or unable to paste lesion. Skipping.")
            continue
        paste_img = (paste_img * 255).astype(np.uint8)
        anatomy_view = target[:, :, 3]
        depth_view = target[:, :, 4]
        depth_img = convert_depth_to_rgb(depth_view)
        view2d.append(paste_img)
        depth2d.append(depth_img)
        anatomy_img = convert_anatomy_to_rgb(anatomy_view)
        anatomy2d.append(anatomy_img)
        mask = target[:, :, 0]
        seg2d.append(mask)
        gr.Info(f"Successfully rendered {img_count+1}/{num_views} image+annotations.")
        img_count += 1
    return view2d, depth2d, anatomy2d, seg2d

# define the list of all the examples
def get_examples():
    # setup_paths()
    # get mesh names from here
    mesh_names = globals()["mesh_names"]
    # get the textures
    textures = ["No Lesion", "Pasted Lesion", "Blended Lesion", "Dilated Lesion"]
    lesions = [1, 2, 5, 10]
    examples = []
    for mesh in mesh_names:
        for texture in textures:
            for lesion in lesions:
                if texture == "No Lesion":
                    # examples.append([mesh, texture, 0, 4, True])
                    examples.append([mesh, texture, 0])
                    break
                # examples.append([mesh, texture, lesion, 4, True])
                examples.append([mesh, texture, lesion])
    return examples


import tempfile


def get_trimesh_attrs(mesh_name, tex_name, num_lesion=1):
    mesh_path = get_mesh_path(mesh_name)
    texture_path = get_texture_module(tex_name)(mesh_name, num_lesion)
    texture_img = Image.open(texture_path).convert("RGB")
    tri_mesh = trimesh.load(mesh_path)

    angle = -math.pi / 2
    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = tf.rotation_matrix(angle, direction, center)
    tri_mesh = tri_mesh.apply_transform(rot_matrix)
    tri_mesh.apply_transform(tf.rotation_matrix(math.pi, [0, 0, 1], [-1, -1, -1]))

    verts, faces = tri_mesh.vertices, tri_mesh.faces
    uvs = tri_mesh.visual.uv
    material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
    vis = trimesh.visual.TextureVisuals(uv=uvs, material=material, image=texture_img)
    tri_mesh.visual = vis
    colors = tri_mesh.visual.to_color()
    vc = colors.vertex_colors  # / 255.0
    # timg = tri_mesh.visual.material.image

    return verts, faces, vc, mesh_name


def plotly_image(image):
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    fig.update_layout(
        width=view_width,
        height=view_height,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(hoverinfo="none")
    return fig


def plotly_mesh(verts, faces, vc, mesh_name):
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vc,
            )
        ]
    )
    # fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False)))
    fig.update_layout(scene=dict(zaxis=dict(visible=False)))
    fig.update_layout(scene=dict(camera=dict(up=dict(x=1, y=0, z=1))))
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=-2, y=-2, z=-1))))
    # disable hover info
    fig.update_traces(hoverinfo="none")
    return fig


def load_texture_map(mesh, mesh_name, texture_name, num_lesion, device=DEVICE):
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    normals = mesh.verts_normals_packed().detach().cpu().numpy()
    texture_path = get_texture_module(texture_name)(mesh_name, num_lesion)
    texture_img = Image.open(texture_path).convert("RGB")
    texture_tensor = torch.from_numpy(np.array(texture_img)).to(DEVICE)
    tmap = TexturesUV(
        maps=texture_tensor.float().to(device=mesh.device).unsqueeze(0),
        verts_uvs=mesh.textures.verts_uvs_padded(),
        faces_uvs=mesh.textures.faces_uvs_padded(),
    )
    return tmap


def load_mesh_and_texture(mesh_name, texture_name, num_lesion=1, device=DEVICE):
    """
    Load a mesh and its corresponding texture.

    Args:
        mesh_name (str): The name of the mesh.
        texture_name (str): The name of the texture module.
        num_lesion (int, optional): The number of lesions. Defaults to 1.
        device (torch.device, optional): The device to load the mesh and texture on. Defaults to DEVICE.

    Returns:
        new_mesh (Meshes): The loaded mesh with texture.
    """
    mesh_path = get_mesh_path(mesh_name)
    texture_path = get_texture_module(texture_name)(mesh_name, num_lesion)
    gr.Info("Loading mesh and texture...")
    mesh = load_objs_as_meshes([mesh_path], device=device)
    tmap = load_texture_map(mesh, mesh_name, texture_name, num_lesion, device)
    new_mesh = Meshes(
        verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tmap
    )
    return new_mesh


def setup_cameras(dist, elev, azim, device=DEVICE):
    gr.Info("Setting up cameras...")
    R, T = look_at_view_transform(dist, elev, azim, degrees=True)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30.0, znear=0.01)
    return cameras


def setup_lights(
    light_pos, ambient_color, diffuse_color, specular_color, device=DEVICE
):
    gr.Info("Setting up lights...")
    lights = PointLights(
        device=device,
        location=light_pos,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
    )
    return lights


def setup_materials(shininess, specularity, device=DEVICE):
    gr.Info("Setting up materials...")
    materials = Materials(
        device=device,
        specular_color=specularity,  # [[specularity, specularity, specularity]],
        shininess=shininess.reshape(-1),  # [shininess],
    )
    return materials


def setup_renderer(cameras, lights, materials, device=DEVICE):
    global raster_settings
    raster_settings = RasterizationSettings(
        image_size=128,
        blur_radius=0.0,
        faces_per_pixel=1,
        # max_faces_per_bin=100,
        # bin_size=0,
        perspective_correct=True,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, materials=materials
        ),
    )
    return renderer


def render_images(renderer, mesh, lights, cameras, materials, nviews, device=DEVICE):
    meshes = mesh.extend(nviews)
    gr.Info("Rendering Images...")
    images = renderer(meshes, lights=lights, cameras=cameras, materials=materials)
    gr.Info("Successfully rendered images.")
    images = images[..., :3]
    images = (images - images.min()) / (images.max() - images.min())
    return images
    fragments = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)(meshes)
    # print(images.shape)
    # breakpoint()
    return images


def randomize_view_params(randomize, num_views):
    dist = torch.rand(num_views).uniform_(0.0, 10.0)
    elev = torch.rand(num_views).uniform_(-90, 90)
    azim = torch.rand(num_views).uniform_(-90, 90)
    light_pos = torch.rand(num_views, 3).uniform_(0.0, 2.0)
    light_ac = torch.rand(num_views, 3).uniform_(0.0, 1.0)
    light_dc = torch.rand(num_views, 3).uniform_(0.0, 1.0)
    light_sc = torch.rand(num_views, 3).uniform_(0.0, 1.0)
    mat_sh = torch.rand(num_views, 1).uniform_(0, 100)
    mat_sc = torch.rand(num_views, 3).uniform_(0.0, 1.0)
    gr.Info("Randomized view parameters...")
    return (
        dist,
        elev,
        azim,
        light_pos,
        light_ac,
        light_dc,
        light_sc,
        mat_sh,
        mat_sc,
    )


def sample_camera_params(num_views, dist, elev, azim):
    gr.Info("Setting up cameras...")
    dist = torch.linspace(dist - num_views // 2, dist + num_views // 2, num_views)
    elev = torch.linspace(elev - num_views // 2, elev + num_views // 2, num_views)
    azim = torch.linspace(azim - num_views // 2, azim + num_views // 2, num_views)
    cameras = setup_cameras(dist, elev, azim)

    return cameras


def sample_light_params(num_views, light_pos, light_ac, light_dc, light_sc):
    gr.Info("Setting up lights...")
    light_pos = (
        torch.linspace(
            light_pos - num_views // 2, light_pos + num_views // 2, num_views
        )
        .reshape(-1, 1)
        .repeat(1, 3)
    )
    light_ac = (
        torch.linspace(light_ac - num_views // 2, light_ac + num_views // 2, num_views)
        .reshape(-1, 1)
        .repeat(1, 3)
    )
    light_dc = (
        torch.linspace(light_dc - num_views // 2, light_dc + num_views // 2, num_views)
        .reshape(-1, 1)
        .repeat(1, 3)
    )
    light_sc = (
        torch.linspace(light_sc - num_views // 2, light_sc + num_views // 2, num_views)
        .reshape(-1, 1)
        .repeat(1, 3)
    )
    lights = setup_lights(light_pos, light_ac, light_dc, light_sc)
    return lights


def sample_material_params(num_views, mat_sh, mat_sc):
    gr.Info("Setting up materials...")
    mat_sh = (
        torch.linspace(mat_sh - num_views // 2, mat_sh + num_views // 2, num_views)
        .reshape(-1, 1)
        .repeat(1, 1)
    )
    mat_sc = (
        torch.linspace(mat_sc - num_views // 2, mat_sc + num_views // 2, num_views)
        .reshape(-1, 1)
        .repeat(1, 3)
    )
    materials = setup_materials(mat_sh, mat_sc)
    return materials


def set_rendering_params(
    randomize,
    num_views,
    dist,
    elev,
    azim,
    light_pos,
    light_ac,
    light_dc,
    light_sc,
    mat_sh,
    mat_sc,
):
    if randomize:
        (
            dist,
            elev,
            azim,
            light_pos,
            light_ac,
            light_dc,
            light_sc,
            mat_sh,
            mat_sc,
        ) = randomize_view_params(randomize, num_views)
        cameras = setup_cameras(dist, elev, azim)
        lights = setup_lights(light_pos, light_ac, light_dc, light_sc)
        materials = setup_materials(mat_sh, mat_sc)
    else:
        cameras = sample_camera_params(num_views, dist, elev, azim)
        lights = sample_light_params(num_views, light_pos, light_ac, light_dc, light_sc)
        materials = sample_material_params(num_views, mat_sh, mat_sc)

    renderer = setup_renderer(cameras, lights, materials)
    return renderer, cameras, lights, materials


def process_examples(mesh_name, tex_name, n_lesion):
    mesh_path = get_mesh_path(mesh_name)
    texture_path = get_texture_module(tex_name)(mesh_name, n_lesion)
    mesh_to_view = plotly_mesh(*get_trimesh_attrs(mesh_name, tex_name, n_lesion))
    return mesh_to_view, texture_path, n_lesion


def update_plots(mesh_name, texture_name, num_lesion):
    if num_lesion > 0 and texture_name == "No Lesion":
        gr.Warning(
            f"Cannot display '{texture_name}' texture map with  {num_lesion} lesions! Please change the texture. Meanwhile, not updating the display."
        )
        return default_mesh_plot, default_texture, num_lesion
    elif num_lesion == 0 and texture_name != "No Lesion":
        go.Warning(
            f"Cannot display '{texture_name}' texture map with {num_lesion} lesions! Please increase the number of lesions."
        )
        return default_mesh_plot, default_texture, num_lesion
    mesh_path = get_mesh_path(mesh_name)
    texture_path = Image.open(get_texture_module(texture_name)(mesh_name, num_lesion)).convert("RGB").resize((512, 512))
    mesh_to_view = plotly_mesh(*get_trimesh_attrs(mesh_name, texture_name, num_lesion))
    gr.Info("Successfully updated mesh and texture.")
    return mesh_to_view, texture_path, num_lesion


def run_demo():
    # get the defined examples
    all_examples = get_examples()

    mesh_block = gr.Plot(
        label="Selected Mesh",
        value=default_mesh_plot,
        # scale=1,
    )
    texture_block = gr.Image(
        value=default_texture,
        type="pil",
        image_mode="RGB",
        height="auto",
        width="auto",
        label="Selected Texture",
    )
    num_lesions = gr.Radio(
        choices=[0, 1, 2, 5, 10],
        label="Number of Lesions",
        value=0,
        interactive=True,
    )
    num_views = gr.Slider(2, 32, 4, label="Number of Views", step=2, interactive=True)
    randomize = gr.Checkbox(
        label="Randomize View Parameters", value=True, interactive=True
    )
    render_button = gr.Button("Render Views")

    select_mesh = gr.Dropdown(
        choices=mesh_names,
        value=mesh_names[0],
        interactive=True,
        label="Input Mesh",
        info="Select the mesh to render",
    )
    select_texture = gr.Dropdown(
        choices=["No Lesion", "Pasted Lesion", "Blended Lesion", "Dilated Lesion"],
        value="No Lesion",
        interactive=True,
        label="Input Texture",
        info="Select the texture to use for the mesh.",
    )
    # compose demo layout and data flow
    with gr.Blocks(
        title=_TITLE, analytics_enabled=True, theme=gr.themes.Base()
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"# {_TITLE}")
            gr.Markdown(_DESCRIPTION)

        # User input panel
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                select_mesh.render()
                select_texture.render()
                num_lesions.render()
                num_views.render()
                randomize.render()

            with gr.Column(scale=1):
                mesh_block.render()
            with gr.Column(scale=1):
                texture_block.render()

            gr.on(
                triggers=[
                    select_mesh.change,
                    select_texture.change,
                    num_lesions.change,
                ],
                inputs=[select_mesh, select_texture, num_lesions],
                outputs=[mesh_block, texture_block, num_lesions],
                fn=update_plots,
            )

            # @gr.on(
            #     inputs=[
            #         select_mesh,
            #         select_texture,
            #         num_lesions,
            #     ],
            #     outputs=[
            #         mesh_block,
            #         texture_block,
            #         num_lesions,
            #     ],
            #     triggers=[
            #         select_mesh.change,
            #         select_texture.change,
            #         num_lesions.change,
            #     ],
            # )
            # def update(m, t, l):
            #     return update_plots(m, t, l)

        # rendering choices
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                render_button.render()
            with gr.Column(scale=1):
                with gr.Accordion("Configure View Parameters", open=False):
                    # setup cameras
                    with gr.Accordion("Camera Parameters", open=False):
                        dist = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=0.5,
                            step=0.5,
                            interactive=True,
                            label="Distance",
                        )
                        elev = gr.Slider(
                            label="Elevation",
                            interactive=True,
                            minimum=-90,
                            maximum=90,
                            value=0,
                            step=10,
                        )
                        azim = gr.Slider(
                            label="Azimuth",
                            interactive=True,
                            minimum=-90,
                            maximum=90,
                            value=90,
                            step=10,
                        )
                    # setup lights
                    with gr.Accordion("Lighting Parameters", open=False):
                        light_pos = gr.Slider(
                            label="Light Position",
                            interactive=True,
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.1,
                        )
                        light_ac = gr.Slider(
                            label="Ambient Color",
                            minimum=0.0,
                            maximum=1.0,
                            interactive=True,
                            value=0.5,
                            step=0.1,
                        )
                        light_dc = gr.Slider(
                            label="Diffuse Color",
                            minimum=0.0,
                            maximum=1.0,
                            interactive=True,
                            value=0.5,
                            step=0.1,
                        )
                        light_sc = gr.Slider(
                            label="Specular Color",
                            minimum=0.0,
                            maximum=1.0,
                            interactive=True,
                            value=0.5,
                            step=0.1,
                        )
                    # setup material parameters
                    with gr.Accordion("Material Parameters", open=False):
                        mat_sh = gr.Slider(
                            label="Shininess",
                            interactive=True,
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=10,
                        )
                        mat_sc = gr.Slider(
                            label="Specularity",
                            minimum=0.0,
                            interactive=True,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                        )

                    update_view_btn = gr.Button("Update View Parameters")

        gr.on(
            triggers=[
                update_view_btn.click,
                dist.change,
                elev.change,
                azim.change,
                light_pos.change,
                light_ac.change,
                light_dc.change,
                light_sc.change,
                mat_sh.change,
                mat_sc.change,
            ],
            inputs=[randomize],
            outputs=[randomize],
            fn=lambda x: False,
            show_progress="hidden",
            queue=False,
            scroll_to_output=True,
        )
        # rendered views panel
        with gr.Row(variant="panel"):
            with gr.Tab("Rendered RGB Views"):
                render_block = gr.Gallery(
                    label="Rendered Views", columns=4, height="auto", object_fit="contain"
                )
            with gr.Tab("Rendered Depth Views"):
                depth_block = gr.Gallery(
                    label="Depth Maps", columns=4, height="auto", object_fit="contain"
                )
            with gr.Tab("Rendered Anatomy Views"):
                anatomy_block = gr.Gallery(
                    label="Anatomy Labels", columns=4, height="auto", object_fit="contain"
                )
            with gr.Tab("Rendered Segmentation Views"):
                seg_block = gr.Gallery(
                    label="Segmentation Masks", columns=4, height="auto", object_fit="contain"
                )
            #
            # render_block = gr.Gallery(
            #     label="Rendered Views", columns=4, height="auto", object_fit="contain"
            # )

        @gr.on(
            triggers=[render_button.click],
            inputs=[
                randomize,
                select_mesh,
                select_texture,
                num_lesions,
                num_views,
                dist,
                elev,
                azim,
                light_pos,
                light_ac,
                light_dc,
                light_sc,
                mat_sh,
                mat_sc,
            ],
            outputs=[render_block, depth_block, anatomy_block, seg_block],
        )
        def render_views(
            randomize,
            select_mesh,
            select_texture,
            num_lesions,
            num_views,
            dist,
            elev,
            azim,
            light_pos,
            light_ac,
            light_dc,
            light_sc,
            mat_sh,
            mat_sc,
        ):
            renderer, cameras, lights, materials = set_rendering_params(
                randomize,
                num_views,
                dist,
                elev,
                azim,
                light_pos,
                light_ac,
                light_dc,
                light_sc,
                mat_sh,
                mat_sc,
            )
            # gr.Info("Loading mesh and texture...")
            # mesh = load_mesh_and_texture(select_mesh, select_texture, num_lesions)
            # cameras
            # images = render_images(
            #     renderer, mesh, lights, cameras, materials, num_views
            # )
            # return [_ for _ in images.detach().cpu().numpy()]
            view2d, depth, anatomy, segmentation = prepare_ds_renderer(
                randomize,
                select_mesh,
                select_texture,
                num_lesions,
                num_views,
                dist,
                elev,
                azim,
                light_pos,
                light_ac,
                light_dc,
                light_sc,
                mat_sh,
                mat_sc,
            )
            return view2d, depth, anatomy, segmentation

        # examples panel when the iuser does not want to input
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                gr.Examples(
                    examples=all_examples,
                    inputs=[
                        select_mesh,
                        select_texture,
                        num_lesions,
                    ],
                    outputs=[
                        mesh_block,
                        texture_block,
                        num_lesions,
                    ],
                    cache_examples=False,
                    fn=update_plots,
                    label="Meshes and Textures for Demo (Click to start)",
                )

    demo.queue(max_size=10)
    demo.launch(
        share=True,
        max_threads=mp.cpu_count(),
        show_error=True,
        show_api=False,
    )


def get_texture_module(tex_type):
    if tex_type == "No Lesion":
        return get_no_lesion_path
    elif tex_type == "Pasted Lesion":
        return get_pasted_lesion_path
    elif tex_type == "Blended Lesion":
        return get_blended_lesion_path
    elif tex_type == "Dilated Lesion":
        return get_dilated_lesion_path
    else:
        raise ValueError(f"Texture type {tex_type} not supported!")


if __name__ == "__main__":
    # setup_paths()
    mesh_paths = glob("./DermSynth3D/data/3dbodytex-1.1-highres/*/*.obj")
    mesh_names = [os.path.basename(os.path.dirname(x)) for x in mesh_paths]
    # get the textures
    all_textures = glob("./DermSynth3D/data/3dbodytex-1.1-highres/*/*.png")
    dir_blended_textures = "./hf_demo/lesions/"
    dir_anatomy = "./DermSynth3D/data/bodytex_anatomy_labels/"
    dir_background = "./DermSynth3D/data/background/IndoorScene/"
    get_no_lesion_path = lambda x, y: os.path.join(
        "./DermSynth3D/data/3dbodytex-1.1-highres", x, "model_highres_0_normalized.png"
    )
    get_mesh_path = lambda x: os.path.join(
        "./DermSynth3D/data/3dbodytex-1.1-highres", x, "model_highres_0_normalized.obj"
    )
    # get the textures with the lesions
    get_mask_path = lambda x: os.path.join(
        "./hf_demo/lesions/", x, "model_highres_0_normalized_mask.png"
    )
    get_dilated_lesion_path = lambda x, y: os.path.join(
        "./hf_demo/lesions/",
        x,
        f"model_highres_0_normalized_dilated_lesion_{y}.png",
    )
    get_blended_lesion_path = lambda x, y: os.path.join(
        "./hf_demo/lesions/",
        x,
        f"model_highres_0_normalized_blended_lesion_{y}.png",
    )
    get_pasted_lesion_path = lambda x, y: os.path.join(
        "./hf_demo/lesions/",
        x,
        f"model_highres_0_normalized_pasted_lesion_{y}.png",
    )
    default_mesh_plot = plotly_mesh(*get_trimesh_attrs(mesh_names[0], "No Lesion", 0))
    default_texture = Image.open(all_textures[0]).convert("RGB").resize((512, 512))
    new_values = {
        "default_mesh_plot": default_mesh_plot,
        "default_texture": default_texture,
    }
    globals().update(new_values)
    run_demo()
