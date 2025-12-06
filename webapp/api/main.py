import pandas as pd
import io
import base64
import json
import trimesh
import os
import sys
import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, Query, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# --- sys.path modifications to import original libraries ---
# Get the absolute path to the project root (DermSynth3D_dev)
PROJECT_ROOT_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT_ABS)
# Add skin3d's parent directory if it's not directly under PROJECT_ROOT_ABS
sys.path.insert(0, os.path.join(PROJECT_ROOT_ABS, 'skin3d'))

# Import the new simplified rendering module
from . import rendering

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Constants ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Data paths - using ../data/ relative to api directory
DATA_DIR = os.path.join(PROJECT_ROOT_ABS, "data")
# use data from github repo
DATA_DIR = "https://github.com/sinashish/DermSynth3D_dev/raw/main/data"

BODYTEX_HIGHRES_DIR = os.path.join(DATA_DIR, "3dbodytex-1.1-highres")
PROCESSED_TEXTURES_DIR = os.path.join(DATA_DIR, "processed_textures")
ANATOMY_LABELS_DIR = os.path.join(DATA_DIR, "bodytex_anatomy_labels")

# --- Helper to load ID to Name Map ---
def _load_id_to_name_map():
    bodytex_csv_path = os.path.join(PROJECT_ROOT_ABS, "skin3d/data/3dbodytex-1.1-highres/bodytex.csv")
    if not os.path.exists(bodytex_csv_path):
        print(f"Warning: bodytex.csv not found at {bodytex_csv_path}. Mesh filtering might be incomplete.")
        return {}
    df = pd.read_csv(bodytex_csv_path, converters={"scan_id": lambda x: str(x)})
    return df.set_index('scan_id')['scan_name'].to_dict()

ID_TO_NAME_MAP = _load_id_to_name_map()

# --- Utility functions from gradio_app.py for output conversion ---
COLOR_LABELS = {
    0: (0., 0., 0.), 1: (174., 199., 232.), 2: (152., 223., 138.),
    3: (31., 119., 180.), 4: (255., 187., 120.), 5: (188., 189., 34.),
    6: (140., 86., 75.), 7: (255., 152., 150.),
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
    if np.all(depth == 0):
        return np.full((depth.shape[0], depth.shape[1], 3), 255, dtype=np.uint8)
    
    # Ensure depth is 2D (H, W)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(axis=-1)
    elif depth.ndim != 2:
        raise ValueError(f"Expected depth to be 2D or 3D (H,W,1), but got shape {depth.shape}")

    mask = depth != 0 # mask is (H, W)
    disp_map = np.zeros_like(depth, dtype=float) # disp_map is (H, W)
    
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
    # This ensures the assignment works correctly across all color channels
    three_channel_mask = np.stack([~mask, ~mask, ~mask], axis=-1)
    colormapped_im[three_channel_mask] = 255 # Assign white to masked out areas
    return colormapped_im


# --- API Endpoints ---

@app.get("/api/config")
def get_config():
    bodytex_csv_path = os.path.join(PROJECT_ROOT_ABS, "skin3d/data/3dbodytex-1.1-highres/bodytex.csv")
    try:
        if not os.path.exists(bodytex_csv_path):
            raise HTTPException(status_code=404, detail="bodytex.csv not found at the expected local path.")
            
        df = pd.read_csv(bodytex_csv_path, converters={"scan_id": lambda x: str(x)})
        
        all_scan_ids = df["scan_id"].unique().tolist()

        available_mesh_ids = []
        for scan_id in all_scan_ids:
            scan_name = ID_TO_NAME_MAP.get(scan_id)
            if scan_name: # Ensure scan_name exists for this id
                # Use scan_name to construct the path for existence check
                obj_path = os.path.join(BODYTEX_HIGHRES_DIR, scan_name, "model_highres_0_normalized.obj")
                if os.path.exists(obj_path):
                    available_mesh_ids.append(scan_id)
        
        mesh_names = sorted(available_mesh_ids)
        texture_names = ["No Lesion", "Pasted Lesion", "Blended Lesion", "Dilated Lesion"]
        return JSONResponse(content={"meshes": mesh_names, "textures": texture_names})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/api/preview-data/{mesh_name}")
def get_preview_data(mesh_name: str, texture_name: str = Query("No Lesion"), num_lesions: int = Query(0)):
    if num_lesions > 0 and texture_name == "No Lesion":
        raise HTTPException(status_code=400, detail="Invalid configuration: 'No Lesion' texture cannot be used with > 0 lesions.")
    if num_lesions == 0 and texture_name != "No Lesion":
        raise HTTPException(status_code=400, detail=f"Invalid configuration: '{texture_name}' requires > 0 lesions.")

    scan_name = ID_TO_NAME_MAP.get(mesh_name)
    if not scan_name:
        raise HTTPException(status_code=404, detail=f"Mesh ID '{mesh_name}' not found in mapping.")

    obj_path = os.path.join(BODYTEX_HIGHRES_DIR, scan_name, "model_highres_0_normalized.obj")
    
    # Determine texture path based on type
    if texture_name == "No Lesion":
        texture_path = os.path.join(BODYTEX_HIGHRES_DIR, scan_name, "model_highres_0_normalized.png")
    else:
        lesion_type_map = {
            "Pasted Lesion": "pasted",
            "Blended Lesion": "blended",
            "Dilated Lesion": "dilated"
        }
        lesion_str = lesion_type_map.get(texture_name, "blended")
        texture_path = os.path.join(PROCESSED_TEXTURES_DIR, scan_name, f"model_highres_0_normalized_{lesion_str}_lesion_{num_lesions}.png")


    try:
        if not os.path.exists(obj_path):
            raise HTTPException(status_code=404, detail=f"Mesh file not found locally for {scan_name} (ID: {mesh_name})")
        if not os.path.exists(texture_path):
            raise HTTPException(status_code=404, detail=f"Texture '{texture_name}' not found locally for mesh {scan_name} (ID: {mesh_name}) with {num_lesions} lesions.")

        with open(obj_path, 'r') as f:
            obj_data = f.read()
        with open(texture_path, 'rb') as f:
            texture_data_bytes = f.read()
        
        texture_data = base64.b64encode(texture_data_bytes).decode("utf-8")
        return JSONResponse(content={"obj_data": obj_data, "texture_data": texture_data})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

@app.websocket("/ws/render")
async def websocket_render(websocket: WebSocket):
    await websocket.accept()
    try:
        while True: # Keep connection open for multiple render jobs
            payload_str = await websocket.receive_text()
            params = json.loads(payload_str)

            mesh_id = params["mesh_name"] # This is the scan_id
            texture_name = params["texture_name"]
            num_lesions = int(params["num_lesions"])
            num_views = int(params["num_views"])
            randomize_views = params["randomize"] # Get randomize flag

            scan_name = ID_TO_NAME_MAP.get(mesh_id)
            if not scan_name:
                await websocket.send_json({"error": f"Mesh ID '{mesh_id}' not found in mapping."})
                continue

            # --- 1. Initialize SimpleRenderer ---
            await websocket.send_json({"status": "Initializing rendering pipeline..."})
            
            mesh_filename = os.path.join(BODYTEX_HIGHRES_DIR, scan_name, "model_highres_0_normalized.obj")
            mesh_dir = os.path.dirname(mesh_filename)
            
            # Get texture path
            texture_path = rendering.get_texture_path(
                PROCESSED_TEXTURES_DIR,
                scan_name,
                texture_name,
                num_lesions,
                mesh_dir,
            )
            
            # Get anatomy labels path
            anatomy_labels_path = rendering.get_anatomy_labels_path(
                ANATOMY_LABELS_DIR,
                scan_name,
            )
            
            # Try to get lesion and nonskin mask paths (optional)
            lesion_mask_path = None
            nonskin_mask_path = None
            if texture_name != "No Lesion" and num_lesions > 0:
                # Try to find lesion mask - may not exist
                lesion_mask_path = os.path.join(
                    PROCESSED_TEXTURES_DIR, scan_name, 
                    f"lesion_mask_lesion_{num_lesions}.png"
                )
                await websocket.send_json({"status": f"Found Lesion mask path"})
                if not os.path.exists(lesion_mask_path):
                    lesion_mask_path = None
                await websocket.send_json({"warning": f"Count not find lesion mask"})
            
            # Try to find nonskin mask - may not exist
            nonskin_mask_path = os.path.join(
                BODYTEX_HIGHRES_DIR, scan_name, "model_highres_0_normalized_mask.png"
            )
            
            if not os.path.exists(nonskin_mask_path):
                nonskin_mask_path = None
                await websocket.send_json({"warning": f"Count not find nonskin mask"})
            if nonskin_mask_path:
                await websocket.send_json({"status": f"Found nonskin mask"})
            
            # Initialize renderer
            renderer = rendering.SimpleRenderer(
                mesh_filename=mesh_filename,
                texture_path=texture_path,
                anatomy_labels_path=anatomy_labels_path,
                device=DEVICE,
                lesion_texture_mask_path=lesion_mask_path,
                nonskin_texture_mask_path=nonskin_mask_path,
            )

            # --- 2. Rendering Loop ---
            import pytorch3d
            import trimesh.proximity
            import trimesh
            
            for i in range(num_views):
                await websocket.send_json({"status": f"Rendering view {i+1} of {num_views}..."})
                
                # Randomize or use fixed camera parameters
                if randomize_views:
                    # Randomize camera position
                    mesh_tri = trimesh.load(mesh_filename, process=False)
                    coords, normals = pytorch3d.ops.sample_points_from_meshes(
                        meshes=renderer.mesh,
                        num_samples=1,
                        return_normals=True,
                    )
                    look_at = coords.cpu().detach().numpy().squeeze()
                    look_at_normal = normals.cpu().detach().numpy().squeeze()
                    
                    normal_weight = np.random.uniform(0.3, 1.5)
                    camera_pos = look_at + look_at_normal * normal_weight
                    
                    # Check if camera is outside mesh
                    if trimesh.proximity.signed_distance(mesh_tri, [camera_pos]) >= 0:
                        await websocket.send_json({"status": f"Skipping view {i+1}: Invalid camera position."})
                        continue
                    
                    dist = 1.0
                    elev = np.random.uniform(-90, 90)
                    azim = np.random.uniform(-90, 90)
                    light_pos = camera_pos + np.random.uniform(-2, 2, 3)
                    ambient = [np.round(np.random.uniform(0.3, 0.99), 2)] * 3
                    specular = [np.round(np.random.uniform(0, 0.1), 2)] * 3
                    diffuse = [np.round(np.random.uniform(0.3, 0.99), 2)] * 3
                    shininess = [np.round(np.random.uniform(30, 60), 2)]
                else:
                    # Fixed camera parameters
                    dist = 1.0
                    elev = 0.0
                    azim = 0.0
                    light_pos = [[0.0, 0.0, -3.0]]
                    ambient = [[0.5, 0.5, 0.5]]
                    specular = [[0.0, 0.0, 0.0]]
                    diffuse = [[0.5, 0.5, 0.5]]
                    shininess = [64]
                
                # Set camera and render
                renderer.set_camera_and_render(
                    dist=dist,
                    elev=elev,
                    azim=azim,
                    view_size=(224, 224),
                    light_location=[light_pos.tolist()] if isinstance(light_pos, np.ndarray) else light_pos,
                    ambient_color=ambient,
                    diffuse_color=diffuse,
                    specular_color=specular,
                    shininess=shininess,
                )
                
                # Extract light parameters for segmentation rendering
                light_loc = light_pos.tolist() if isinstance(light_pos, np.ndarray) else light_pos[0] if isinstance(light_pos[0], list) else light_pos
                amb = tuple(ambient[0]) if isinstance(ambient[0], list) else tuple(ambient)
                spec = tuple(specular[0]) if isinstance(specular[0], list) else tuple(specular)
                diff = tuple(diffuse[0]) if isinstance(diffuse[0], list) else tuple(diffuse)
                shin = shininess[0] if isinstance(shininess, list) else shininess
                
                # Render all maps
                rgb, depth_view, anatomy_view, seg_view = renderer.render_all(
                    light_location=light_loc,
                    ambient=amb,
                    specular=spec,
                    diffuse=diff,
                    shininess=shin,
                )

                def encode_image(img_array, mode='RGB'):
                    # Ensure img_array is float and scaled to 0-1 before converting to uint8
                    if img_array.dtype != np.float32:
                        img_array = img_array.astype(np.float32)
                    if img_array.max() > 1.0:
                        img_array = img_array / 255.0 # Scale if it's 0-255
                    
                    img = Image.fromarray((img_array * 255).astype(np.uint8), mode)
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")

                rgb_b64 = encode_image(rgb)
                
                # Ensure anatomy_view is 2D and convert to uint8 (make a copy to avoid modifying target)
                anatomy_view = np.asarray(anatomy_view)  # Make a copy
                if anatomy_view.ndim > 2:
                    anatomy_view = anatomy_view.squeeze()
                if anatomy_view.ndim != 2:
                    raise ValueError(f"Expected anatomy_view to be 2D, but got shape {anatomy_view.shape}")
                # Convert to uint8 for label indices (anatomy labels are integers)
                anatomy_view = anatomy_view.astype(np.uint8)
                
                # Ensure depth_view is 2D and convert to float (make a copy to avoid modifying target)
                depth_view = np.asarray(depth_view)  # Make a copy
                if depth_view.ndim > 2:
                    depth_view = depth_view.squeeze()
                if depth_view.ndim != 2:
                    raise ValueError(f"Expected depth_view to be 2D, but got shape {depth_view.shape}")
                # Convert to float32 for depth values
                depth_view = depth_view.astype(np.float32)
                
                depth_pil = Image.fromarray(rendering.convert_depth_to_rgb(depth_view))
                anatomy_pil = Image.fromarray(rendering.convert_anatomy_to_rgb(anatomy_view))
                
                with io.BytesIO() as buffer:
                    depth_pil.save(buffer, format="PNG")
                    depth_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                with io.BytesIO() as buffer:
                    anatomy_pil.save(buffer, format="PNG")
                    anatomy_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # For segmentation, take one channel from the 3-channel paste_mask
                seg_b64 = encode_image(seg_view[:,:,], mode='RGB')

                await websocket.send_json({
                    "image": rgb_b64,
                    "depth": depth_b64,
                    "anatomy": anatomy_b64,
                    "segmentation": seg_b64,
                    "status": f"Completed view {i+1} of {num_views}."
                })

            await websocket.send_json({"status": "Render complete. Ready for next job."})

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        try:
            await websocket.send_json({"error": error_message})
        except:
            pass

# --- Root and Static File Routes ---
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

app.mount("/static", StaticFiles(directory="static"), name="static")
