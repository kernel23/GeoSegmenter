# app.py - The Local SAM Backend Server (Optimized for Large Files & Advanced Prompts)

import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor
import base64
import json # For parsing GeoJSON
import shapefile # For shapefile export
import io # For in-memory file handling
import zipfile # For zipping shapefile components
import os # For path joining, though less needed with in-memory
from PIL import Image
import rasterio
from rasterio.transform import Affine
import ee # Import Earth Engine API
import requests # For downloading image from URL

print("--- Starting Georeferenced Server (Large File Optimized) ---")

print("\nINFO: To use Google Earth Engine functionality, you might need to authenticate.")
print("Please run 'earthengine authenticate' in your terminal if you haven't already.\n")

# --- 1. Configuration ---
CURRENT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Initial device: {CURRENT_DEVICE}")
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

# Global model variables that might be reloaded
sam = None
predictor = None

# Resolution mapping (width, height) - assuming landscape, adjust if needed or make dynamic
RESOLUTION_MAPPING = {
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
    "original": None # Special case for original resolution
}

# --- 2. Model Loading Function ---
def load_model(device):
    global sam, predictor
    print(f"Loading SAM model '{MODEL_TYPE}' onto device: {device}...")
    try:
        sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam_model.to(device=device)
        predictor_instance = SamPredictor(sam_model)
        print("Model loaded successfully.")
        return sam_model, predictor_instance
    except FileNotFoundError:
        print(f"ERROR: Model checkpoint not found at '{CHECKPOINT_PATH}'")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        # Potentially fall back to CPU or exit
        if device != torch.device('cpu'):
            print("Attempting to load model on CPU as a fallback...")
            return load_model(torch.device('cpu'))
        else:
            exit()

sam, predictor = load_model(CURRENT_DEVICE)

# --- 3. Flask App Initialization ---
app = Flask(__name__)
CORS(app)
app.state = {
    'image_shape': None, # Full original image shape
    'model_input_shape': None, # Shape of image fed to SAM
    'transform': None,
    'preview_shape': None,
    'current_resolution_key': 'original', # Default
    'image_source_type': None, # To track if image is from file, EE, or snapshot
    'current_snapshot_gcp_data': None # To hold GCPs for a snapshot
}

# --- 4. API Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    global CURRENT_DEVICE
    return jsonify({
        'status': 'ok',
        'message': 'SAM georeferenced backend is running.',
        'current_device': str(CURRENT_DEVICE),
        'available_resolutions': list(RESOLUTION_MAPPING.keys())
    })

@app.route('/set_device', methods=['POST'])
def set_device():
    global CURRENT_DEVICE, sam, predictor
    json_data = request.get_json()
    device_preference = json_data.get('device_preference', '').lower()

    new_device = None
    if device_preference == 'gpu':
        if torch.cuda.is_available():
            new_device = torch.device('cuda:0')
        else:
            return jsonify({'error': 'GPU (CUDA) not available on the server.'}), 400
    elif device_preference == 'cpu':
        new_device = torch.device('cpu')
    else:
        return jsonify({'error': 'Invalid device preference. Choose "cpu" or "gpu".'}), 400

    if new_device == CURRENT_DEVICE:
        return jsonify({'message': f'Device is already set to {CURRENT_DEVICE}. No change needed.'}), 200

    print(f"Switching device from {CURRENT_DEVICE} to {new_device}...")
    try:
        sam, predictor = load_model(new_device)
        CURRENT_DEVICE = new_device
        if 'original_cv2_image' in app.state and app.state['original_cv2_image'] is not None:
            print("Re-embedding existing image on new device...")
            image_to_embed = app.state['original_cv2_image']
            resolution_key = app.state.get('current_resolution_key', 'original')
            processed_image_for_sam, model_input_shape = process_image_for_sam(
                app.state['original_cv2_image_full_res'],
                resolution_key
            )
            app.state['model_input_shape'] = model_input_shape
            predictor.set_image(cv2.cvtColor(processed_image_for_sam, cv2.COLOR_BGR2RGB))
            print("Image re-embedded successfully on new device.")
        return jsonify({'message': f'Successfully switched to {CURRENT_DEVICE}. Model reloaded.'}), 200
    except Exception as e:
        print(f"Failed to switch device or reload model: {e}")
        return jsonify({'error': f'Failed to switch device: {str(e)}'}), 500

def process_image_for_sam(cv2_image_full_res, resolution_key):
    full_h, full_w = cv2_image_full_res.shape[:2]
    if resolution_key == "original" or resolution_key not in RESOLUTION_MAPPING or RESOLUTION_MAPPING[resolution_key] is None:
        max_dim_original = 8192
        if max(full_h, full_w) > max_dim_original:
            scale = max_dim_original / max(full_h, full_w)
            target_w, target_h = int(full_w * scale), int(full_h * scale)
            print(f"Original image too large ({full_w}x{full_h}), scaling to {target_w}x{target_h} for SAM.")
            img_for_sam = cv2.resize(cv2_image_full_res, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            img_for_sam = cv2_image_full_res.copy()
    else:
        target_w, target_h = RESOLUTION_MAPPING[resolution_key]
        original_aspect = full_w / full_h
        target_aspect = target_w / target_h
        if original_aspect > target_aspect:
            final_w = target_w
            final_h = int(target_w / original_aspect)
        else:
            final_h = target_h
            final_w = int(target_h * original_aspect)
        print(f"Resizing image for SAM from {full_w}x{full_h} to {final_w}x{final_h} for resolution '{resolution_key}'")
        img_for_sam = cv2.resize(cv2_image_full_res, (final_w, final_h), interpolation=cv2.INTER_AREA)
    return img_for_sam, img_for_sam.shape[:2]

@app.route('/set_image_from_path', methods=['POST'])
def set_image_from_path():
    print("Received request to set image from local path.")
    json_data = request.get_json()
    image_path = json_data.get('image_path')
    world_file_path = json_data.get('world_file_path')
    resolution_key = json_data.get('resolution', 'original')

    if resolution_key not in RESOLUTION_MAPPING:
        return jsonify({'error': f'Invalid resolution key provided: {resolution_key}. Valid keys are: {list(RESOLUTION_MAPPING.keys())}'}), 400

    app.state['current_resolution_key'] = resolution_key
    app.state['image_source_type'] = 'local_file'
    app.state['current_snapshot_gcp_data'] = None # Clear GCP data

    if not image_path or not world_file_path:
        return jsonify({'error': 'Image or world file path not provided'}), 400
    if not os.path.exists(image_path) or not os.path.exists(world_file_path):
        return jsonify({'error': 'One or both file paths do not exist on the server.'}), 400

    try:
        print(f"Loading image from: {image_path}")
        cv2_image_full_res = cv2.imread(image_path)
        if cv2_image_full_res is None:
            raise IOError("Could not read the image file. Check file format and path.")

        app.state['original_cv2_image_full_res'] = cv2_image_full_res
        original_full_shape = cv2_image_full_res.shape[:2]
        app.state['image_shape'] = original_full_shape

        print(f"Loading world file from: {world_file_path}")
        with open(world_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        A_val, D_val, B_val, E_val, C_val, F_val = map(float, lines)
        C_adj = C_val - (A_val / 2.0)
        F_adj = F_val - (E_val / 2.0)
        transform_matrix = Affine(A_val, B_val, C_adj, D_val, E_val, F_adj)
        app.state['transform'] = transform_matrix
        transform_params = [A_val, B_val, C_adj, D_val, E_val, F_adj]
        print(f"Georeferencing transform loaded. Original C,F: ({C_val},{F_val}), Adjusted C,F: ({C_adj},{F_adj})")

        img_for_sam, model_input_shape = process_image_for_sam(cv2_image_full_res, resolution_key)
        app.state['model_input_shape'] = model_input_shape

        preview_max_dim = 2048
        preview_scale_factor = preview_max_dim / max(original_full_shape)
        preview_w = int(original_full_shape[1] * preview_scale_factor)
        preview_h = int(original_full_shape[0] * preview_scale_factor)
        preview_image = cv2.resize(cv2_image_full_res, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        app.state['preview_shape'] = preview_image.shape[:2]

        is_success, buffer = cv2.imencode(".jpg", preview_image)
        if not is_success:
            raise ValueError("Could not encode preview image.")
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        preview_data_url = f"data:image/jpeg;base64,{preview_base64}"

    except Exception as e:
        print(f"Error processing files: {e}")
        return jsonify({'error': str(e)}), 500

    print(f"Setting image (resolution: {resolution_key}, shape: {model_input_shape}) in SAM predictor...")
    predictor.set_image(cv2.cvtColor(img_for_sam, cv2.COLOR_BGR2RGB))
    print("Image embedding is set. Ready for predictions.")

    return jsonify({
        'message': f'Image set successfully with resolution {resolution_key}',
        'preview_image': preview_data_url,
        'geo_transform_params': transform_params,
        'full_image_shape': {'height': original_full_shape[0], 'width': original_full_shape[1]},
        'model_input_shape': {'height': model_input_shape[0], 'width': model_input_shape[1]}
    })

@app.route('/set_image_from_ee', methods=['POST'])
def set_image_from_ee():
    print("Received request to set image from Google Earth Engine.")
    try:
        ee.Initialize()
        print("Earth Engine Initialized successfully.")
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        return jsonify({'error': f'Earth Engine initialization failed: {str(e)}. Have you run "earthengine authenticate"?'}), 500

    json_data = request.get_json()
    ee_image_id = json_data.get('ee_image_id')
    roi_data = json_data.get('roi')
    resolution_key = json_data.get('resolution', 'original')
    bands_str = json_data.get('bands', 'B4,B3,B2')
    date_range_data = json_data.get('date_range')
    vis_min = json_data.get('vis_min')
    vis_max = json_data.get('vis_max')

    if not ee_image_id or not roi_data:
        return jsonify({'error': 'Earth Engine Image ID or ROI not provided'}), 400
    if resolution_key not in RESOLUTION_MAPPING:
        return jsonify({'error': f'Invalid resolution key provided: {resolution_key}.'}), 400

    app.state['current_resolution_key'] = resolution_key
    app.state['image_source_type'] = 'earth_engine'
    app.state['current_snapshot_gcp_data'] = None

    try:
        bands = [b.strip() for b in bands_str.split(',')]
        if len(bands) != 3:
            return jsonify({'error': 'Please provide exactly 3 bands for visualization.'}), 400

        ee_roi = ee.Geometry.Rectangle([roi_data['min_lon'], roi_data['min_lat'], roi_data['max_lon'], roi_data['max_lat']])
        ee_object = ee.ImageCollection(ee_image_id) if "/" in ee_image_id else ee.Image(ee_image_id)

        if isinstance(ee_object, ee.ImageCollection):
            if date_range_data and date_range_data.get('start') and date_range_data.get('end'):
                ee_object = ee_object.filterDate(date_range_data['start'], date_range_data['end'])
            image = ee_object.median().clip(ee_roi)
        else:
            image = ee_object.clip(ee_roi)

        if not image:
            return jsonify({'error': 'Could not load EE image with the given parameters.'}), 400

        current_vis_min = 0
        current_vis_max = 3000
        if vis_min is not None: current_vis_min = float(vis_min)
        if vis_max is not None: current_vis_max = float(vis_max)

        vis_params = {'bands': bands, 'min': current_vis_min, 'max': current_vis_max}
        print(f"Using visualization parameters: {vis_params}")

        thumb_width, thumb_height = 1024, 1024
        params = {'region': ee_roi.toGeoJSONString(), 'format': 'png', 'dimensions': f'{thumb_width}x{thumb_height}'}
        params.update(vis_params)
        thumb_url = image.getThumbURL(params)

        response = requests.get(thumb_url)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download image from EE (status {response.status_code}).'}), 500

        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        cv2_image_full_res = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        app.state['original_cv2_image_full_res'] = cv2_image_full_res
        original_full_shape = cv2_image_full_res.shape[:2]
        app.state['image_shape'] = original_full_shape

        img_width_pixels, img_height_pixels = original_full_shape[1], original_full_shape[0]
        A = (roi_data['max_lon'] - roi_data['min_lon']) / img_width_pixels
        E = (roi_data['min_lat'] - roi_data['max_lat']) / img_height_pixels
        C, F = roi_data['min_lon'], roi_data['max_lat']
        transform_matrix = Affine(A, 0, C, 0, E, F)
        app.state['transform'] = transform_matrix
        transform_params = [A, 0, C, 0, E, F]

        img_for_sam, model_input_shape = process_image_for_sam(cv2_image_full_res, resolution_key)
        app.state['model_input_shape'] = model_input_shape

        preview_max_dim = 2048
        preview_scale_factor = preview_max_dim / max(original_full_shape) if max(original_full_shape) > 0 else 1
        preview_w = int(original_full_shape[1] * preview_scale_factor)
        preview_h = int(original_full_shape[0] * preview_scale_factor)
        preview_image = cv2.resize(cv2_image_full_res, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        app.state['preview_shape'] = preview_image.shape[:2]

        is_success, buffer = cv2.imencode(".jpg", preview_image)
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        preview_data_url = f"data:image/jpeg;base64,{preview_base64}"

    except Exception as e:
        print(f"Error processing Earth Engine image: {e}")
        return jsonify({'error': str(e)}), 500

    print(f"Setting EE image (resolution: {resolution_key}, shape: {model_input_shape}) in SAM predictor...")
    predictor.set_image(cv2.cvtColor(img_for_sam, cv2.COLOR_BGR2RGB))
    print("EE Image embedding is set. Ready for predictions.")

    return jsonify({
        'message': f'Earth Engine image set successfully with resolution {resolution_key}',
        'preview_image': preview_data_url,
        'geo_transform_params': transform_params,
        'full_image_shape': {'height': original_full_shape[0], 'width': original_full_shape[1]},
        'model_input_shape': {'height': model_input_shape[0], 'width': model_input_shape[1]}
    })

@app.route('/set_image_from_snapshot', methods=['POST'])
def set_image_from_snapshot():
    print("Received request to set image from snapshot.")
    json_data = request.get_json()
    snapshot_data_url = json_data.get('snapshot_image')
    resolution_key = json_data.get('resolution', 'original')
    geo_gcps = json_data.get('geo_gcps')
    pixel_gcps = json_data.get('pixel_gcps')

    if not snapshot_data_url:
        return jsonify({'error': 'Snapshot image data not provided'}), 400
    if resolution_key not in RESOLUTION_MAPPING:
        return jsonify({'error': f'Invalid resolution key: {resolution_key}'}), 400

    app.state['current_resolution_key'] = resolution_key
    app.state['current_snapshot_gcp_data'] = None

    valid_gcps_provided = False
    if geo_gcps and pixel_gcps and len(geo_gcps) >= 4 and len(pixel_gcps) >= 4:
        valid_gcps_provided = True

    if valid_gcps_provided:
        app.state['image_source_type'] = 'snapshot_georef'
        app.state['current_snapshot_gcp_data'] = {'geo_gcps': geo_gcps, 'pixel_gcps': pixel_gcps}
        print("Snapshot GCPs received. Image type set to 'snapshot_georef'.")
    else:
        app.state['image_source_type'] = 'snapshot'
        print("No valid GCPs provided. Treating as plain snapshot.")

    try:
        header, base64_str = snapshot_data_url.split(',', 1)
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv2_image_full_res = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        app.state['original_cv2_image_full_res'] = cv2_image_full_res
        original_full_shape = cv2_image_full_res.shape[:2]
        app.state['image_shape'] = original_full_shape
        app.state['transform'] = Affine.identity()
        transform_params = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        img_for_sam, model_input_shape = process_image_for_sam(cv2_image_full_res, resolution_key)
        app.state['model_input_shape'] = model_input_shape

        preview_max_dim = 2048
        if max(original_full_shape) > preview_max_dim:
            scale = preview_max_dim / max(original_full_shape)
            preview_w, preview_h = int(original_full_shape[1] * scale), int(original_full_shape[0] * scale)
            preview_image_cv2 = cv2.resize(cv2_image_full_res, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            app.state['preview_shape'] = preview_image_cv2.shape[:2]
        else:
            preview_image_cv2 = cv2_image_full_res
            app.state['preview_shape'] = original_full_shape

        is_success, buffer = cv2.imencode(".jpg", preview_image_cv2)
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        preview_data_url_for_response = f"data:image/jpeg;base64,{preview_base64}"

    except Exception as e:
        print(f"Error processing snapshot image: {e}")
        return jsonify({'error': str(e)}), 500

    print(f"Setting snapshot image (resolution: {resolution_key}, shape: {model_input_shape}) in SAM predictor...")
    predictor.set_image(cv2.cvtColor(img_for_sam, cv2.COLOR_BGR2RGB))
    print("Snapshot image embedding is set. Ready for predictions.")

    return jsonify({
        'message': f'Snapshot image set successfully with resolution {resolution_key}',
        'preview_image': preview_data_url_for_response,
        'geo_transform_params': transform_params,
        'full_image_shape': {'height': original_full_shape[0], 'width': original_full_shape[1]},
        'model_input_shape': {'height': model_input_shape[0], 'width': model_input_shape[1]},
        'image_source_type': app.state['image_source_type']
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_shape' not in app.state or 'model_input_shape' not in app.state:
        return jsonify({'error': 'Image or model input configuration not set.'}), 400

    json_data = request.get_json()
    preview_h, preview_w = app.state['preview_shape']
    model_h, model_w = app.state['model_input_shape']
    scale_preview_to_model_w = model_w / preview_w
    scale_preview_to_model_h = model_h / preview_h

    input_points, input_labels, input_box = [], [], None

    if 'points' in json_data and json_data['points']:
        for p in json_data['points']:
            input_points.append([int(p['x'] * scale_preview_to_model_w), int(p['y'] * scale_preview_to_model_h)])
            input_labels.append(p['label'])

    if 'box' in json_data and json_data['box']:
        box = json_data['box']
        input_box = np.array([
            int(box['x1'] * scale_preview_to_model_w), int(box['y1'] * scale_preview_to_model_h),
            int(box['x2'] * scale_preview_to_model_w), int(box['y2'] * scale_preview_to_model_h)
        ])

    if not input_points and input_box is None:
        return jsonify({'error': 'No prompts provided.'}), 400

    masks, scores, logits = predictor.predict(
        point_coords=np.array(input_points) if input_points else None,
        point_labels=np.array(input_labels) if input_labels else None,
        box=input_box,
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)] # This mask is relative to model_input_shape

    # --- Create a colored RGBA image of the mask for visualization ---
    # The mask is the size of the image SAM processed (model_input_shape)
    color = np.array([np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(100, 256), 128]) # RGBA with alpha
    h, w = best_mask.shape[-2:]
    mask_image_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_image_rgba[best_mask] = color

    # Encode this colored mask image to send to the frontend
    is_success, buffer = cv2.imencode(".png", mask_image_rgba)
    if not is_success:
        return jsonify({'error': 'Failed to encode mask image'}), 500
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    mask_data_url = f"data:image/png;base64,{mask_base64}"


    # --- Convert mask from model_input_shape to original full_image_shape for georeferencing ---
    original_full_h, original_full_w = app.state['image_shape']
    mask_for_resize = best_mask.astype(np.uint8) * 255
    mask_full_res = cv2.resize(mask_for_resize, (original_full_w, original_full_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_full_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({'polygon': [], 'mask_image': None})

    largest_contour = max(contours, key=cv2.contourArea)
    pixel_polygon_full_res = largest_contour.squeeze().tolist()
    if not isinstance(pixel_polygon_full_res, list) or (pixel_polygon_full_res and not isinstance(pixel_polygon_full_res[0], list)):
        pixel_polygon_full_res = [pixel_polygon_full_res] if not isinstance(pixel_polygon_full_res, list) else [p.tolist() for p in pixel_polygon_full_res]

    image_source = app.state.get('image_source_type')

    # Default response
    response_data = {
        'polygon_type': 'pixel',
        'polygon': pixel_polygon_full_res,
        'mask_image': mask_data_url
    }

    if image_source == 'snapshot_georef':
        gcp_data = app.state.get('current_snapshot_gcp_data')
        if gcp_data and gcp_data.get('pixel_gcps') and gcp_data.get('geo_gcps'):
            pixel_gcps_np = np.array(gcp_data['pixel_gcps'], dtype=np.float32)
            geo_gcps_np = np.array(gcp_data['geo_gcps'], dtype=np.float32)
            polygon_points_np = np.array([pixel_polygon_full_res], dtype=np.float32)
            try:
                H, _ = cv2.findHomography(pixel_gcps_np, geo_gcps_np)
                if H is None:
                    raise ValueError("Homography matrix could not be computed.")
                transformed_polygon_np = cv2.perspectiveTransform(polygon_points_np, H)
                geo_polygon = transformed_polygon_np[0].tolist()
                response_data['polygon_type'] = 'geographic'
                response_data['polygon'] = geo_polygon
            except Exception as e:
                response_data['error_message'] = f'GCP transform failed: {str(e)}'
        else:
            response_data['error_message'] = 'GCP data missing for georeferencing.'

    elif image_source in ['local_file', 'earth_engine']:
        transform_affine = app.state['transform']
        geo_polygon = [transform_affine * point for point in pixel_polygon_full_res]
        response_data['polygon_type'] = 'geographic'
        response_data['polygon'] = geo_polygon

    return jsonify(response_data)


def create_shapefile_components(polygons_list, is_geographic_data):
    shp_buffer, shx_buffer, dbf_buffer = io.BytesIO(), io.BytesIO(), io.BytesIO()
    writer = shapefile.Writer(shp=shp_buffer, shx=shx_buffer, dbf=dbf_buffer)
    writer.shapeType = shapefile.POLYGON

    writer.field('id', 'N', 10)
    writer.field('Farmer_N', 'C', 50)
    writer.field('Farm_Add', 'C', 100)
    writer.field('Reg_Area', 'N', 12, 2)
    writer.field('Tob_Type', 'C', 30)
    writer.field('Sub_Type', 'C', 30)
    writer.field('Variety', 'C', 30)
    writer.field('Val_Area', 'N', 12, 2)
    writer.field('Diff', 'N', 12, 2)
    writer.field('Color', 'C', 30)

    for poly_data in polygons_list:
        points = poly_data.get('points')
        if not points or len(points) < 3: continue
        writer.poly([points])
        writer.record(
            poly_data.get('id', 0), poly_data.get('Farmer_Name', '')[:50],
            poly_data.get('Farm_Address', '')[:100], poly_data.get('Registered_Area', 0.0),
            poly_data.get('Tobacco_Type', '')[:30], poly_data.get('Sub_type', '')[:30],
            poly_data.get('Variety', '')[:30], poly_data.get('Validated_Area', 0.0),
            poly_data.get('Difference', 0.0), poly_data.get('color', '')[:30]
        )
    writer.close()

    prj_content = None
    if is_geographic_data:
        prj_content = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'

    return {'shp': shp_buffer.getvalue(), 'shx': shx_buffer.getvalue(), 'dbf': dbf_buffer.getvalue(), 'prj': prj_content.encode('utf-8') if prj_content else None}

@app.route('/load_external_polygons', methods=['POST'])
def load_external_polygons():
    print("Received request to load external polygons.")
    file = request.files.get('geojson_file') # Use .get to avoid KeyError if not present

    processed_polygons = []
    feature_id_counter = app.state.get('next_external_polygon_id', 10000) 
    default_color = f"rgba({np.random.randint(0,100)}, {np.random.randint(0,100)}, {np.random.randint(200,256)}, 1)"

    if file and file.filename != '':
        if file.filename.endswith('.zip'):
            print("Processing Zip file for Shapefile.")
            try:
                shp_file, dbf_file, prj_file = None, None, None
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    # Find shp, dbf, prj files in zip
                    for member_name in zip_ref.namelist():
                        if member_name.endswith('.shp'):
                            shp_file = io.BytesIO(zip_ref.read(member_name))
                        elif member_name.endswith('.dbf'):
                            dbf_file = io.BytesIO(zip_ref.read(member_name))
                        elif member_name.endswith('.prj'):
                            prj_file = io.BytesIO(zip_ref.read(member_name)) # We are not using .prj content for now
                    
                    if not shp_file or not dbf_file:
                        return jsonify({'error': 'Zip file must contain .shp and .dbf files.'}), 400

                    # Attempt to read Shapefile with multiple encodings for DBF
                    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'cp850']
                    sf = None
                    last_exception = None

                    for enc in encodings_to_try:
                        try:
                            # Need to reset file pointers for each attempt if they were consumed
                            shp_file.seek(0)
                            dbf_file.seek(0)
                            sf = shapefile.Reader(shp=shp_file, dbf=dbf_file, encoding=enc)
                            # Try to access some data to trigger potential decoding errors early
                            _ = sf.fields 
                            _ = sf.shapeRecords()[:1] # Try to read one record
                            print(f"Successfully opened Shapefile with encoding: {enc}")
                            last_exception = None # Clear last exception if successful
                            break 
                        except UnicodeDecodeError as e:
                            print(f"Failed to decode Shapefile with encoding {enc}: {e}")
                            last_exception = e
                            sf = None # Ensure sf is None if an error occurred
                        except shapefile.ShapefileException as e: # Catch other pyshp errors
                            print(f"Pyshp error with encoding {enc}: {e}")
                            # If it's not a decode error, it might be a structural issue with the SHP/DBF
                            # For now, we'll let it be caught by the broader exception handler if all encodings fail.
                            # Or, decide if certain pyshp errors should be fatal immediately.
                            # For simplicity here, we'll assume structural errors are less common than encoding ones.
                            last_exception = e # Store it
                            sf = None
                            break # Break if it's a pyshp error not related to encoding of this attempt.

                    if sf is None:
                        error_msg = f'Failed to read Shapefile DBF with tried encodings: {encodings_to_try}.'
                        if last_exception:
                            error_msg += f' Last error: {str(last_exception)}'
                        return jsonify({'error': error_msg}), 400
                    
                    # Proceed with sf (shapefile.Reader instance)
                    # The try block here was removed as its exceptions are covered by the outer try/except ShapefileException
                    if sf.shapeType != shapefile.POLYGON and sf.shapeType != shapefile.POLYGONZ and sf.shapeType != shapefile.POLYGONM:
                         return jsonify({'error': f'Shapefile contains non-polygon geometries (type: {sf.shapeType}). Only polygons are supported.'}), 400
                    
                    fields = [field[0] for field in sf.fields[1:]] 
                    temp_shapefile_poly_count = 0
                    for shaperec in sf.iterShapeRecords():
                        shape = shaperec.shape
                        record = shaperec.record
                        
                        properties = dict(zip(fields, record))
                        
                        # Assuming parts are outer rings, then inner rings. We only take the first part (outer ring).
                        # And we assume points are [lon, lat]
                        points = shape.points
                        if not points: continue

                        # Ensure polygon is closed by checking if first and last points are the same
                        # pyshp points are usually [(x,y), (x,y)...]
                        formatted_points = [[pt[0], pt[1]] for pt in points]
                        if formatted_points[0][0] != formatted_points[-1][0] or formatted_points[0][1] != formatted_points[-1][1]:
                            formatted_points.append(list(formatted_points[0]))

                        processed_polygons.append({
                            'id': properties.get('id', feature_id_counter), # Use 'id' field from dbf if present
                            'points': formatted_points,
                            'properties': properties,
                            'isGeographic': True, # Assume WGS84 for now
                            'isExternal': True,
                            'sourceType': 'shapefile',
                            'color': default_color 
                        })
                        feature_id_counter += 1
                        temp_shapefile_poly_count += 1
                print(f"Processed {temp_shapefile_poly_count} polygons from Shapefile.")
            except zipfile.BadZipFile:
                return jsonify({'error': 'Invalid or corrupted zip file.'}), 400
            except shapefile.ShapefileException as e:
                return jsonify({'error': f'Error reading Shapefile: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'An unexpected error occurred processing Shapefile: {str(e)}'}), 500

        elif file.filename.endswith('.geojson') or file.filename.endswith('.json') or file.mimetype == 'application/json':
            print("Processing GeoJSON file.")
            try:
                geojson_data = json.load(file)
                # ... (existing GeoJSON processing logic from before) ...
                if geojson_data.get("type") == "FeatureCollection":
                    for feature in geojson_data.get("features", []):
                        geometry = feature.get("geometry")
                        properties = feature.get("properties", {})
                        if geometry and geometry.get("type") == "Polygon":
                            coordinates = geometry.get("coordinates", [])
                            if coordinates and len(coordinates) > 0:
                                external_ring = coordinates[0]
                                if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                                    processed_polygons.append({
                                        'id': properties.get('id', feature_id_counter),
                                        'points': external_ring, 
                                        'properties': properties,
                                        'isGeographic': True, 
                                        'isExternal': True,
                                        'sourceType': 'geojson',
                                        'color': default_color
                                    })
                                    feature_id_counter +=1
                        elif geometry and geometry.get("type") == "MultiPolygon":
                            for polygon_coords in geometry.get("coordinates", []):
                                if polygon_coords and len(polygon_coords) > 0:
                                    external_ring = polygon_coords[0]
                                    if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                                        processed_polygons.append({
                                            'id': properties.get('id', feature_id_counter),
                                            'points': external_ring,
                                            'properties': properties, 
                                            'isGeographic': True,
                                            'isExternal': True,
                                            'sourceType': 'geojson',
                                            'color': default_color
                                        })
                                        feature_id_counter +=1
                elif geojson_data.get("type") == "Feature":
                    geometry = geojson_data.get("geometry")
                    properties = geojson_data.get("properties", {})
                    if geometry and geometry.get("type") == "Polygon":
                        coordinates = geometry.get("coordinates", [])
                        if coordinates and len(coordinates) > 0:
                            external_ring = coordinates[0]
                            if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                                processed_polygons.append({
                                    'id': properties.get('id', feature_id_counter),
                                    'points': external_ring,
                                    'properties': properties,
                                    'isGeographic': True,
                                    'isExternal': True,
                                    'sourceType': 'geojson',
                                    'color': default_color
                                })
                                feature_id_counter +=1
                print(f"Processed {len(processed_polygons)} polygons from GeoJSON file.")
            except Exception as e:
                return jsonify({'error': f'Error parsing GeoJSON file: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a .zip (for Shapefile) or .geojson file.'}), 400

    elif request.is_json: # Handling pasted GeoJSON text
        print("Processing pasted GeoJSON text.")
        json_data_req = request.get_json()
        if 'geojson_data' not in json_data_req:
            return jsonify({'error': 'Missing geojson_data in request body'}), 400
        geojson_data = json_data_req['geojson_data']
        if not isinstance(geojson_data, dict):
             return jsonify({'error': 'geojson_data must be a valid GeoJSON object.'}), 400
        # ... (existing GeoJSON processing logic from before, adapted for geojson_data variable) ...
        if geojson_data.get("type") == "FeatureCollection":
            for feature in geojson_data.get("features", []):
                geometry = feature.get("geometry")
                properties = feature.get("properties", {})
                if geometry and geometry.get("type") == "Polygon":
                    coordinates = geometry.get("coordinates", [])
                    if coordinates and len(coordinates) > 0:
                        external_ring = coordinates[0]
                        if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                            processed_polygons.append({
                                'id': properties.get('id', feature_id_counter),
                                'points': external_ring, 
                                'properties': properties,
                                'isGeographic': True, 
                                'isExternal': True,
                                'sourceType': 'geojson_text',
                                'color': default_color
                            })
                            feature_id_counter +=1
                elif geometry and geometry.get("type") == "MultiPolygon":
                    for polygon_coords in geometry.get("coordinates", []):
                        if polygon_coords and len(polygon_coords) > 0:
                            external_ring = polygon_coords[0]
                            if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                                processed_polygons.append({
                                    'id': properties.get('id', feature_id_counter),
                                    'points': external_ring,
                                    'properties': properties, 
                                    'isGeographic': True,
                                    'isExternal': True,
                                    'sourceType': 'geojson_text',
                                    'color': default_color
                                })
                                feature_id_counter +=1
        elif geojson_data.get("type") == "Feature":
            geometry = geojson_data.get("geometry")
            properties = geojson_data.get("properties", {})
            if geometry and geometry.get("type") == "Polygon":
                coordinates = geometry.get("coordinates", [])
                if coordinates and len(coordinates) > 0:
                    external_ring = coordinates[0]
                    if all(isinstance(pt, list) and len(pt) == 2 for pt in external_ring):
                        processed_polygons.append({
                            'id': properties.get('id', feature_id_counter),
                            'points': external_ring,
                            'properties': properties,
                            'isGeographic': True,
                            'isExternal': True,
                            'sourceType': 'geojson_text',
                            'color': default_color
                        })
                        feature_id_counter +=1
        print(f"Processed {len(processed_polygons)} polygons from pasted GeoJSON text.")
    else:
        return jsonify({'error': 'No file or JSON data provided for external layer.'}), 400

    app.state['next_external_polygon_id'] = feature_id_counter
    
    if not processed_polygons:
        return jsonify({'error': 'No valid Polygon geometries found in the provided data.'}), 400

    print(f"Successfully processed a total of {len(processed_polygons)} external polygons.")

    # Convert to GeoJSON FeatureCollection for map display
    layer_name_for_geojson = "External Layer"
    if file and file.filename:
        layer_name_for_geojson = file.filename
    
    geojson_feature_collection = convertToGeoJSONFeatureCollection(processed_polygons, layer_name_for_geojson)

    return jsonify({
        'message': f'{len(processed_polygons)} external polygons loaded successfully.',
        'external_polygons': processed_polygons, # For canvas drawing and existing logic
        'geojson_for_map_data': geojson_feature_collection # For Google Maps data layer
    }), 200

def convertToGeoJSONFeatureCollection(polygons_list, default_layer_name="External Layer"):
    features = []
    for poly_idx, poly_data in enumerate(polygons_list):
        # Ensure points are correctly formatted as [[[lon, lat], ...]] for GeoJSON Polygon
        # poly_data['points'] is expected to be like [[lon, lat], [lon, lat], ...] for the outer ring
        coordinates = [poly_data.get('points', [])] # GeoJSON Polygon coordinates are list of rings

        feature_properties = {
            # Include original properties from DBF or GeoJSON feature
            **(poly_data.get('properties') or {}), 
            # Add/override with our application-specific data
            'internal_id': poly_data.get('id', f"poly_{poly_idx}"), # Use internal ID
            'color': poly_data.get('color', 'rgba(0,0,255,0.5)'), # Default or assigned color
            'sourceType': poly_data.get('sourceType', 'unknown'),
            'layerName': poly_data.get('layerName', default_layer_name) # Add layer name if available, else default
        }
        
        # Clean up None properties, GeoJSON doesn't like null property values sometimes
        cleaned_properties = {k: v for k, v in feature_properties.items() if v is not None}


        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates 
            },
            "properties": cleaned_properties
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.route('/download_shapefile', methods=['POST'])
def download_shapefile():
    try:
        polygons_data = request.get_json()
        if not polygons_data:
            return jsonify({'error': 'Invalid or missing polygon data.'}), 400

        geographic_polygons = [p for p in polygons_data if p.get('isGeographic', False)]
        pixel_polygons = [p for p in polygons_data if not p.get('isGeographic', False)]

        if not geographic_polygons and not pixel_polygons:
            return jsonify({'error': 'No polygons to export.'}), 400

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if geographic_polygons:
                geo_components = create_shapefile_components(geographic_polygons, True)
                for ext, data in geo_components.items():
                    if data: zf.writestr(f'geographic/geo_polygons.{ext}', data)
            if pixel_polygons:
                pixel_components = create_shapefile_components(pixel_polygons, False)
                for ext, data in pixel_components.items():
                    if data: zf.writestr(f'pixel/pixel_polygons.{ext}', data)

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name='polygons_shapefiles.zip', mimetype='application/zip')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate shapefiles: {str(e)}'}), 500

if __name__ == '__main__':
    if sam is None or predictor is None:
        sam, predictor = load_model(CURRENT_DEVICE)
    if sam is None:
        print("Failed to load model. Exiting.")
        exit()
    app.run(port=5000, debug=True)
