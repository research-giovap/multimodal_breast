"""
Pre-processing and geometry utilities of LesionLoop Breast.

- DICOM loading, with conversion of PET to body-weight SUV;
- rigid alignment of PET to the CT grid;
- resampling to the inference grid (1.171875 x 1.171875 x 3.0 mm), CT windowing (40/400 HU)
  and PET min-max normalisation;
- extraction of the 2.5D patches (3 CT + 3 PET slices + prompt map, 224 x 224);
- mapping of 2D and 3D predictions back to the CT geometry.
"""

import logging
import os

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom import dcmread
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.transform import resize

log = logging.getLogger(__name__)

# Allow pydicom to convert wrong-length elements to UN
pydicom.config.convert_wrong_length_to_UN = True


def extract_tag(ds, tag):
    """
    Recursively search for a DICOM tag inside the dataset and nested sequences.
    Example: extract_tag(ds, (0x0018, 0x1074))  # Radiopharmaceutical Total Dose
    """
    if tag in ds:
        return ds[tag].value
    for elem in ds:
        if elem.VR == 'SQ':  # Sequence
            for item in elem.value:
                val = extract_tag(item, tag)
                if val is not None:
                    return val
    return None


def get_image(path):
    """
    Load a DICOM image or series robustly, returning:
    - SimpleITK image (in SUV if PET, otherwise original)
    - Numpy array (in SUV if PET, otherwise original)
    - Modality
    - Full metadata dictionary (excluding pixel data)
    """

    # --- Load SimpleITK image ---
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        sitk_image = reader.Execute()
    else:
        dicom_names = [path]
        sitk_image = sitk.ReadImage(path)

    numpy_image = sitk.GetArrayFromImage(sitk_image).astype(float)
    sitk_image_suv = sitk_image  # default fallback

    # --- Collect metadata ---
    full_metadata = {}
    pixel_tags = {'7FE0|0010'}  # PixelData
    for f in dicom_names:
        ds = dcmread(f, force=True)
        for elem in ds.iterall():
            tag_str = f"{elem.tag.group:04X}|{elem.tag.elem:04X}"
            if tag_str in pixel_tags or tag_str in full_metadata:
                continue
            try:
                full_metadata[tag_str] = elem.value
            except Exception as e:
                log.debug("Skipping tag %s due to a parsing error: %s", tag_str, e)

    modality = full_metadata.get('0008|0060', None)

    # --- PET conversion to SUVbw ---
    if modality == "PT":
        try:
            ds_sample = dcmread(dicom_names[0], force=True)

            # --- Extract PET metadata ---
            weight_kg = extract_tag(ds_sample, (0x0010, 0x1030))  # Patient Weight
            injected_dose_bq = extract_tag(ds_sample, (0x0018, 0x1074))  # Total Dose
            half_life_s = extract_tag(ds_sample, (0x0018, 0x1075))  # Half-life
            scan_time_str = extract_tag(ds_sample, (0x0008, 0x0031))  # Series Time
            injection_time_str = extract_tag(ds_sample, (0x0018, 0x1072))  # Start Time

            # --- If injected dose missing or 0.0, check deeper sequences ---
            if injected_dose_bq in [None, 0, 0.0, "0", "0.0", ""]:
                rph_seq = extract_tag(ds_sample, (0x0054, 0x0016))  # Radiopharmaceutical Information Sequence
                if rph_seq and hasattr(rph_seq, "__iter__"):
                    try:
                        injected_dose_bq = extract_tag(rph_seq[0], (0x0018, 0x1074))
                    except Exception:
                        injected_dose_bq = 0

            # --- Safe conversion to float ---
            def safe_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            weight_kg = safe_float(weight_kg)
            injected_dose_bq = safe_float(injected_dose_bq)
            half_life_s = safe_float(half_life_s)
            scan_time_str = str(scan_time_str or "000000")
            injection_time_str = str(injection_time_str or "000000")

            log.debug("SUV parameters: weight %s kg, injected dose %s Bq, half-life %s s, "
                      "series time %s, injection time %s",
                      weight_kg, injected_dose_bq, half_life_s, scan_time_str, injection_time_str)

            # --- Validation ---
            if injected_dose_bq == 0 or weight_kg == 0 or half_life_s == 0:
                raise ValueError("Missing essential PET metadata for SUV conversion (Dose, Weight, or HalfLife).")

            # --- Convert HHMMSS to seconds ---
            def hhmmss_to_seconds(hhmmss):
                hhmmss = str(hhmmss).split('.')[0].zfill(6)
                h = int(hhmmss[0:2]); m = int(hhmmss[2:4]); s = int(hhmmss[4:6])
                return h*3600 + m*60 + s

            elapsed_seconds = hhmmss_to_seconds(scan_time_str) - hhmmss_to_seconds(injection_time_str)
            if elapsed_seconds < 0:
                log.warning("Series time precedes injection time; no decay correction applied.")
                elapsed_seconds = 0

            # --- Dose decay correction ---
            dose_at_acq = injected_dose_bq * (2 ** (-elapsed_seconds / half_life_s))
            if dose_at_acq == 0:
                raise ValueError("Dose at acquisition time is zero, cannot compute SUV.")

            # --- SUVbw calculation ---
            suv_factor = (weight_kg * 1000) / dose_at_acq
            numpy_image = numpy_image * suv_factor

            sitk_image_suv = sitk.GetImageFromArray(numpy_image)
            sitk_image_suv.CopyInformation(sitk_image)

            log.info("PET converted to body-weight SUV.")

        except Exception as e:
            log.warning("Could not convert PET to SUVbw (%s); returning the original values.", e)

    return sitk_image_suv, numpy_image, modality, full_metadata


def get_sitk_interpolator(method_name: str) -> int:
    """Maps a string name to a SimpleITK interpolator enumeration."""
    if method_name.lower() == 'linear':
        return sitk.sitkLinear
    elif method_name.lower() == 'bspline':
        return sitk.sitkBSpline
    elif method_name.lower() == 'nearest':
        return sitk.sitkNearestNeighbor
    else:
        log.warning("Unknown interpolator '%s'; using linear.", method_name)
        return sitk.sitkLinear


def resample(sitk_image, np_image, new_spacing, interpolation_method='nearest', return_sitk_obj=False):
    if interpolation_method =='nearest':
        order = 0   # Nearest neighbor
    elif interpolation_method =='linear':
        order = 1   # Trilinear
    elif interpolation_method =='bspline':
        order = 3   # Cubic B-spline
    else:
        log.warning("Unknown interpolation method '%s'; using nearest neighbour.", interpolation_method)
        order = 0   # Nearest neighbor
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    new_size = [
    int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
    for i in range(3)
    ]
   
    resized_volume = resize(
        np_image,
        output_shape=new_size[::-1],  # Note: skimage uses (z, y, x) order
        order=order,                  # Use trilinear interpolation
        preserve_range=True,      # Crucial: Keep the original data range!
        anti_aliasing=True        # Recommended for downsampling to prevent artifacts
        )
    if return_sitk_obj:
        resized_sitk = sitk.GetImageFromArray(resized_volume)
        resized_sitk.SetSpacing(new_spacing)
        resized_sitk.SetOrigin(sitk_image.GetOrigin())
        resized_sitk.SetDirection(sitk_image.GetDirection())
        return resized_sitk
        
    return resized_volume


def align_pet_to_ct(pet_image: sitk.Image, ct_image: sitk.Image, interpolation_method: str = 'linear'):
    """
    Aligns and resamples a PET image to the CT's spatial frame with a robust check for Z-axis orientation.
    """
    
    # --- 1. ROBUSTNESS CHECK (Perform BEFORE alignment) ---
    # Get the Z-component of the Z-axis vector from the direction matrix for both images.
    # This is the 9th element (index 8).
    original_pet_z_dir = pet_image.GetDirection()[8]
    reference_ct_z_dir = ct_image.GetDirection()[8]

    # Determine if a flip is needed for display. This occurs if their Z-axes point in opposite directions.
    # np.sign() returns -1 for negative, 1 for positive, 0 for zero.
    needs_z_flip_for_display = np.sign(original_pet_z_dir) != np.sign(reference_ct_z_dir)
    
    if needs_z_flip_for_display:
        log.info("PET and CT have opposite z-axis directions; the aligned PET array will be flipped.")

    log.debug("CT size %s, origin %s, direction %s, spacing %s", ct_image.GetSize(), ct_image.GetOrigin(),
              ct_image.GetDirection(), ct_image.GetSpacing())
    log.debug("PET size %s, origin %s, direction %s, spacing %s", pet_image.GetSize(), pet_image.GetOrigin(),
              pet_image.GetDirection(), pet_image.GetSpacing())

    ct_image = sitk.Cast(ct_image, sitk.sitkFloat64)
    pet_image = sitk.Cast(pet_image, sitk.sitkFloat64)

    # --- 2. ALIGNMENT ---
    initial_transform = sitk.CenteredTransformInitializer(
        ct_image,
        pet_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(get_sitk_interpolator(interpolation_method))
    resampler.SetTransform(initial_transform)
    resampler.SetDefaultPixelValue(0)

    resampled_pet_image = resampler.Execute(pet_image)
    resampled_pet_array = sitk.GetArrayFromImage(resampled_pet_image)

    # --- 3. CONDITIONAL FLIP (Perform AFTER alignment) ---
    # Apply the Z-axis flip only if our initial check determined it was necessary.
    if needs_z_flip_for_display:
        resampled_pet_array = resampled_pet_array[::-1, :, :]

    log.debug("Aligned PET size %s, spacing %s", resampled_pet_image.GetSize(), resampled_pet_image.GetSpacing())

    return resampled_pet_image, resampled_pet_array


def apply_window(volume, center=40, width=400):
    lower = center - width/2
    upper = center + width/2
    vol_win = np.clip(volume, lower, upper)
    vol_norm = (vol_win - lower) / (upper - lower)
    return vol_norm  # now in [0, 1]


def min_max_normalize(volume):
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val == 0:
        return np.zeros_like(volume)  # Avoid division by zero
    vol_norm = (volume - min_val) / (max_val - min_val)
    return vol_norm  # now in [0, 1]


def extract_25d_patch_centered(vol, center_z, center_y, center_x, patch_size=224):
    """Extracts a 2.5D patch (3, H, W) centered at the given coordinates."""
    depth, height, width = vol.shape
    half_size = patch_size // 2

    # Y and X bounds
    y_start, y_end = center_y - half_size, center_y + half_size
    x_start, x_end = center_x - half_size, center_x + half_size

    # Z indices (z-1, z, z+1) with clamping
    slice_indices = [
        max(0, min(center_z - 1, depth - 1)),
        max(0, min(center_z,     depth - 1)),
        max(0, min(center_z + 1, depth - 1))
    ]

    patch_volume = np.zeros((3, patch_size, patch_size), dtype=vol.dtype)

    for i, z_idx in enumerate(slice_indices):
        full_slice = vol[z_idx, :, :]
        
        # Calculate valid overlap
        img_y_start, img_y_end = max(0, y_start), min(height, y_end)
        img_x_start, img_x_end = max(0, x_start), min(width, x_end)

        patch_y_start = max(0, -y_start)
        patch_y_end = patch_y_start + (img_y_end - img_y_start)
        patch_x_start = max(0, -x_start)
        patch_x_end = patch_x_start + (img_x_end - img_x_start)

        if img_y_end > img_y_start and img_x_end > img_x_start:
            patch_volume[i, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
                full_slice[img_y_start:img_y_end, img_x_start:img_x_end]

    return patch_volume


def generate_prompt_map(mask_2d, sigma=20):
    """
    Generates a prompt map using Exponential Distance Transform.
    Formula: exp(-distance / sigma)
    """
    # Ensure binary (float32)
    scribble_binary = (mask_2d > 0).astype(np.float32)
    
    # If mask is empty, distance is infinite (or max), resulting in 0 map.
    # If mask is full, distance is 0, resulting in 1.
    if not np.any(scribble_binary):
        return np.zeros_like(mask_2d, dtype=np.float32)

    # Calculate Euclidean distance to the nearest ZERO pixel (background)
    # distance_transform_edt(input) calculates distance to nearest BACKGROUND (0).
    # So we calculate distance on (1 - scribble).
    distance_map = distance_transform_edt(1 - scribble_binary)
    
    prompt_map = np.exp(-distance_map / sigma)
    return prompt_map.astype(np.float32)


def prepare_volumes_for_inference(sitk_ct, numpy_ct, sitk_pet_aligned, numpy_pet_aligned):
    """
    Run once after loading: prepares the volumes on the inference grid.
    - Resamples volumes to target spacing (1.17, 1.17, 3.0).
    - Windows CT and Normalizes PET.
    - Returns the scale factors and the reference geometry used to map results back.
    """
    target_spacing = (1.171875, 1.171875, 3.0)

    # 1. Resample CT (Keep SITK object for reference)
    sitk_ct_resampled = resample(sitk_ct, numpy_ct, target_spacing, 'linear', return_sitk_obj=True)
    resampled_np_ct = sitk.GetArrayFromImage(sitk_ct_resampled)
    
    # 2. Resample PET (Only need numpy array for inference)
    resampled_np_pet = resample(sitk_pet_aligned, numpy_pet_aligned, target_spacing, 'linear')
    
    # 3. Preprocess (Windowing / Normalization)
    vol_ct_final = apply_window(resampled_np_ct, center=40, width=400)
    vol_pet_final = min_max_normalize(resampled_np_pet)
    
    # 4. Calculate Scale Factors (To map clicks from Viewer -> Resampled Volume)
    # Viewer Shape (Z, Y, X) vs Resampled Shape (Z, Y, X)
    orig_shape = numpy_ct.shape
    new_shape = resampled_np_ct.shape
    
    scale_info = {
        'scale_z': new_shape[0] / orig_shape[0],
        'scale_y': new_shape[1] / orig_shape[1],
        'scale_x': new_shape[2] / orig_shape[2],
        'sitk_resampled_ref': sitk_ct_resampled, # Critical for reconstruction
        'resampled_shape': new_shape
    }

    return vol_ct_final, vol_pet_final, scale_info


def run_inference_on_preprocessed(vol_ct_final, vol_pet_final, scale_info, mask_input_viewer_2d, viewer_slice_idx):
    """
    Builds the network input for a prompt drawn in the viewer:
    - Maps the operator's click or scribble to the inference grid.
    - Extracts the 2.5D CT and PET patches centred on the prompt.
    - Encodes the prompt as an exponential distance map.
    """
    # 1. Map Coordinates (Viewer -> Resampled)
    z_viewer = viewer_slice_idx
    z_resampled = int(round(z_viewer * scale_info['scale_z']))
    z_resampled = min(max(0, z_resampled), vol_ct_final.shape[0] - 1)

    # 2. Map the 2D Mask (Resize the scribble to match resampled resolution)
    # This replaces 'resample(sitk_mask...)' for the single slice
    resampled_h, resampled_w = vol_ct_final.shape[1], vol_ct_final.shape[2]
    
    if np.sum(mask_input_viewer_2d) > 0:
        # Use skimage resize (Nearest Neighbor) to preserve binary scribble
        mask_resampled_2d = resize(
            mask_input_viewer_2d, 
            (resampled_h, resampled_w), 
            order=0, 
            preserve_range=True, 
            anti_aliasing=False
        )
    else:
        mask_resampled_2d = np.zeros((resampled_h, resampled_w), dtype=np.float32)

    # 3. Centroid of the prompt
    if np.sum(mask_resampled_2d) > 0:
        coords = center_of_mass(mask_resampled_2d > 0)
        cy, cx = int(round(coords[0])), int(round(coords[1]))
    else:
        cy, cx = resampled_h // 2, resampled_w // 2
    
    cz = z_resampled # The Z is determined by the slice we are on

    # 4. Extract patches
    patch_ct = extract_25d_patch_centered(vol_ct_final, cz, cy, cx, 224)
    patch_pet = extract_25d_patch_centered(vol_pet_final, cz, cy, cx, 224)
    
    # Crop the prompt with the same bounds as extract_25d_patch_centered
    patch_size = 224
    half_size = patch_size // 2
    y_start, y_end = cy - half_size, cy + half_size
    x_start, x_end = cx - half_size, cx + half_size
    
    img_y_start, img_y_end = max(0, y_start), min(resampled_h, y_end)
    img_x_start, img_x_end = max(0, x_start), min(resampled_w, x_end)
    
    patch_y_start = max(0, -y_start)
    patch_y_end = patch_y_start + (img_y_end - img_y_start)
    patch_x_start = max(0, -x_start)
    patch_x_end = patch_x_start + (img_x_end - img_x_start)
    
    center_mask_slice = np.zeros((patch_size, patch_size), dtype=np.float32)
    if img_y_end > img_y_start and img_x_end > img_x_start:
         center_mask_slice[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
            mask_resampled_2d[img_y_start:img_y_end, img_x_start:img_x_end]

    # 5. Prompt map
    prompt_map_2d = generate_prompt_map(center_mask_slice, sigma=20)
    prompt_map_channel = np.expand_dims(prompt_map_2d, axis=0)

    # 6. Stack
    final_input_volume = np.concatenate([patch_ct, patch_pet, prompt_map_channel], axis=0)

    # 7. Prepare Metadata for Reconstruction
    metadata = {
        'centroid': (cz, cy, cx),
        'shape': scale_info['resampled_shape'],
        'sitk_ref': scale_info['sitk_resampled_ref']
    }
    
    return final_input_volume, metadata


def reconstruct_and_resample_back_optimized(pred_patch_2d, metadata, original_sitk_reference):
    """
    Maps a 2D prediction back to the viewer:
    1. Places the 224x224 prediction patch into the empty Resampled Volume.
    2. Assigns spatial metadata (Spacing 1.17, Origin, etc.) to that volume.
    3. Resamples that volume back to the Original CT Geometry (Viewer Space).
    """

    # 1. Unpack Metadata from the Inference Step
    cz, cy, cx = metadata['centroid']
    resampled_shape = metadata['shape']      # Shape of the Resampled Volume (Z, Y, X)
    sitk_resampled_ref = metadata['sitk_ref'] # The SimpleITK object defining Resampled Geometry
    
    # 2. Reconstruct the volume in Resampled Space
    # Create an empty volume with the dimensions of the RESAMPLED CT
    reconstructed_vol = np.zeros(resampled_shape, dtype=np.uint8)
    
    # Logic to paste the patch (Inverse of extract_25d_patch_centered)
    patch_h, patch_w = pred_patch_2d.shape
    half_size = patch_h // 2
    
    # Calculate bounds in the Resampled Volume
    y_start, y_end = cy - half_size, cy + half_size
    x_start, x_end = cx - half_size, cx + half_size
    
    # Handle Volume Boundaries (Clamping)
    img_y_start, img_y_end = max(0, y_start), min(resampled_shape[1], y_end)
    img_x_start, img_x_end = max(0, x_start), min(resampled_shape[2], x_end)
    
    # Handle Patch Boundaries (Clamping)
    patch_y_start = max(0, -y_start)
    patch_y_end = patch_y_start + (img_y_end - img_y_start)
    patch_x_start = max(0, -x_start)
    patch_x_end = patch_x_start + (img_x_end - img_x_start)
    
    # Paste the prediction
    if img_y_end > img_y_start and img_x_end > img_x_start:
        reconstructed_vol[cz, img_y_start:img_y_end, img_x_start:img_x_end] = \
            pred_patch_2d[patch_y_start:patch_y_end, patch_x_start:patch_x_end]

    # 3. Create SimpleITK Image & Assign Spatial Info
    # This turns the numpy array into a spatially aware "Segmentation Object"
    seg_sitk = sitk.GetImageFromArray(reconstructed_vol)
    seg_sitk.CopyInformation(sitk_resampled_ref) # geometry of the inference grid

    # 4. Resample BACK to Viewer Space (Original CT Geometry)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_sitk_reference) # target: original CT geometry
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest Neighbor for Binary Masks
    resampler.SetDefaultPixelValue(0)
    
    # The transform is Identity because the spatial coordinates (Origin/Direction) 
    # handles the alignment automatically.
    final_seg_sitk = resampler.Execute(seg_sitk)
    
    # 5. Return Numpy Array (matches self.segmentation_mask shape)
    final_seg_numpy = sitk.GetArrayFromImage(final_seg_sitk)
    
    return final_seg_numpy


def run_inference_direct_ai_space(vol_ct_final, vol_pet_final, mask_input_2d, z_ai):
    """
    Inference helper for the propagation loop.
    Unlike the main inference function, this works PURELY in AI-Space (Resampled coordinates).
    It does not perform coordinate mapping from the viewer.
    """

    d, h, w = vol_ct_final.shape
    
    # 1. Calculate Centroid of the input mask (the previous slice's prediction)
    if np.sum(mask_input_2d) > 0:
        coords = center_of_mass(mask_input_2d > 0)
        cy, cx = int(round(coords[0])), int(round(coords[1]))
    else:
        # Fallback if mask is empty (should ideally break loop before this)
        cy, cx = h // 2, w // 2 

    # 2. Extract Patches (Directly using AI coords)
    patch_ct = extract_25d_patch_centered(vol_ct_final, z_ai, cy, cx, 224)
    patch_pet = extract_25d_patch_centered(vol_pet_final, z_ai, cy, cx, 224)
    
    # 3. Extract/Crop Mask Patch
    patch_size = 224
    half = patch_size // 2
    
    y_min, y_max = max(0, cy - half), min(h, cy + half)
    x_min, x_max = max(0, cx - half), min(w, cx + half)
    
    patch_y_min = max(0, half - cy)
    patch_y_max = patch_y_min + (y_max - y_min)
    patch_x_min = max(0, half - cx)
    patch_x_max = patch_x_min + (x_max - x_min)
    
    center_mask_slice = np.zeros((patch_size, patch_size), dtype=np.float32)
    if y_max > y_min and x_max > x_min:
        center_mask_slice[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = \
            mask_input_2d[y_min:y_max, x_min:x_max]

    # 4. Prompt Map
    prompt_map_2d = generate_prompt_map(center_mask_slice, sigma=20)
    prompt_map_channel = np.expand_dims(prompt_map_2d, axis=0)

    # 5. Stack
    input_vol = np.concatenate([patch_ct, patch_pet, prompt_map_channel], axis=0)
    
    metadata = {
        'centroid_ai': (z_ai, cy, cx),
        'shape': (d, h, w)
    }
    
    return input_vol, metadata


def paste_patch_into_slice(pred_patch, metadata):
    """
    Reconstructs a single 2D slice from the prediction patch into the full AI-space frame.
    """
    z, cy, cx = metadata['centroid_ai']
    full_shape = metadata['shape']
    
    full_slice = np.zeros((full_shape[1], full_shape[2]), dtype=np.uint8)
    
    patch_h, patch_w = pred_patch.shape
    half = patch_h // 2
    
    y_min, y_max = max(0, cy - half), min(full_shape[1], cy + half)
    x_min, x_max = max(0, cx - half), min(full_shape[2], cx + half)
    
    py_min = max(0, half - cy)
    py_max = py_min + (y_max - y_min)
    px_min = max(0, half - cx)
    px_max = px_min + (x_max - x_min)
    
    if y_max > y_min and x_max > x_min:
        full_slice[y_min:y_max, x_min:x_max] = pred_patch[py_min:py_max, px_min:px_max]
        
    return full_slice


def find_suv_max_centroid(vol_pet, binary_mask_3d):
    """
    Finds the (Z, Y, X) coordinates of the maximum intensity pixel 
    WITHIN the binary_mask_3d.
    """
    # Mask the PET volume
    masked_pet = np.where(binary_mask_3d > 0, vol_pet, -np.inf)
    
    # Find flattened index of max value
    if np.all(masked_pet == -np.inf):
        return None
        
    flat_idx = np.argmax(masked_pet)
    z, y, x = np.unravel_index(flat_idx, vol_pet.shape)
    
    return int(z), int(y), int(x)


def create_local_seed_mask(shape, center, radius=3):
    """
    Creates a small circular seed at the specific center.
    """
    z, cy, cx = center
    d, h, w = shape
    
    mask_2d = np.zeros((h, w), dtype=np.uint8)
    
    y, x = np.ogrid[:h, :w]
    dist_sq = (y - cy)**2 + (x - cx)**2
    mask_2d[dist_sq <= radius**2] = 1
    
    return mask_2d


def resample_full_mask_back_to_viewer(ai_mask_3d, sitk_ref_ai, sitk_ref_viewer):
    """
    Takes the final 3D AI mask, gives it spatial info, and resamples 
    it back to the original Viewer geometry.
    """
    
    # 1. Turn numpy array into SITK Image
    seg_sitk = sitk.GetImageFromArray(ai_mask_3d)
    seg_sitk.CopyInformation(sitk_ref_ai) # Assign the 1.17mm spacing info
    
    # 2. Resample to Viewer
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_ref_viewer)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Binary -> Nearest
    resampler.SetDefaultPixelValue(0)
    
    final_sitk = resampler.Execute(seg_sitk)
    return sitk.GetArrayFromImage(final_sitk)


def keep_largest_connected_component_3d(vol_mask):
    """
    ROBUSTNESS: 3D Connectivity.
    Removes floating artifacts/islands. Keeps only the largest connected volume.
    """
    from skimage.measure import label
    
    if np.sum(vol_mask) == 0: return vol_mask
    
    # Label all 3D blobs (connected components)
    labeled_vol, num_features = label(vol_mask, return_num=True)
    
    if num_features <= 1:
        return vol_mask
    
    # Iterate to find the largest blob
    largest_label = 0
    largest_size = 0
    
    # Note: labels start from 1 (0 is background)
    for i in range(1, num_features + 1):
        size = np.sum(labeled_vol == i)
        if size > largest_size:
            largest_size = size
            largest_label = i
            
    # Return binary mask of only the largest blob
    return (labeled_vol == largest_label).astype(np.uint8)


def filter_disconnected_blobs(prev_mask, curr_mask):
    """
    ROBUSTNESS: Active Filtering with Tolerance.
    1. Identify all blobs in 'curr_mask'.
    2. Check if each blob touches the 'prev_mask' (or is very close).
    3. Keep ONLY the touching blobs. Discard new islands.
    """
    from skimage.measure import label
    from skimage.morphology import binary_dilation, disk
    
    # If mask is empty, return empty
    if np.sum(curr_mask) == 0:
        return curr_mask
        
    # 1. Label distinct blobs in the new mask
    labeled_mask, num_features = label(curr_mask, return_num=True)
    
    if num_features == 0:
        return curr_mask
        
    # Prepare the "Tolerance Zone" (Dilate previous mask by 3 pixels)
    # This allows the tumor to shift slightly without being deleted.
    prev_dilated = binary_dilation(prev_mask, disk(3))
    
    filtered_mask = np.zeros_like(curr_mask)
    
    # 2. Iterate through each blob
    for i in range(1, num_features + 1):
        current_blob = (labeled_mask == i).astype(np.uint8)
        
        # 3. Check Overlap with the TOLERANCE zone
        # We use 'prev_dilated' instead of 'prev_mask' to be safe against diagonal shifts
        intersection = np.logical_and(prev_dilated, current_blob)
        
        if np.sum(intersection) > 0:
            # KEEP: It connects to the previous structure
            filtered_mask = np.logical_or(filtered_mask, current_blob)
        else:
            # DISCARD: It is a new, floating island
            pass
            
    return filtered_mask.astype(np.uint8)
