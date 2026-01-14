import os
import SimpleITK as sitk
import numpy as np
from pydicom import dcmread
import pydicom
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.morphology import binary_erosion, disk, binary_dilation, remove_small_objects, binary_closing, label
from scipy.ndimage import binary_fill_holes
import networkx as netx
from skimage.morphology import skeletonize
import random
import math
from scipy.ndimage import distance_transform_edt, center_of_mass
from skimage.draw import line, ellipse


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
                print(f"Warning: skipping tag {tag_str} due to parsing error: {e}")

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

            # --- Logging for debug ---
            print(f"weight: {weight_kg}")
            print(f"injected_dose: {injected_dose_bq}")
            print(f"half_life: {half_life_s}")
            print(f"scan_time: {scan_time_str}")
            print(f"injection_time: {injection_time_str}")

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
                print("Warning: Scan time appears before injection time — assuming no decay.")
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

            print("✅ PET successfully converted to SUVbw.")

        except Exception as e:
            print(f"Warning: Could not convert PET to SUVbw: {e}. Returning image with original values.")

    return sitk_image_suv, numpy_image, modality, full_metadata

def get_mask(path):
    mask_image = sitk.ReadImage(path)

    # Convert to a NumPy array if needed
    mask_array = sitk.GetArrayFromImage(mask_image)
    return mask_image, mask_array

def get_sitk_interpolator(method_name: str) -> int:
    """Maps a string name to a SimpleITK interpolator enumeration."""
    if method_name.lower() == 'linear':
        return sitk.sitkLinear
    elif method_name.lower() == 'bspline':
        return sitk.sitkBSpline
    elif method_name.lower() == 'nearest':
        return sitk.sitkNearestNeighbor
    else:
        print(f"Warning: Unknown interpolator '{method_name}'. Defaulting to linear.")
        return sitk.sitkLinear

def physical_resize(np_image: np.ndarray, pixel_spacing: tuple, interpolation_method: str = 'nearest'):
    """Resize a 3D NumPy array to a new shape using specified interpolation.
    """
    print(f"start resizing volume...")
    if interpolation_method =='nearest':
        order = 0   # Nearest neighbor
    elif interpolation_method =='linear':
        order = 1   # Trilinear
    elif interpolation_method =='bspline':
        order = 3   # Cubic B-spline
    else:
        print(f"Warning: Unknown interpolation method '{interpolation_method}'. Defaulting to nearest neighbor.")
        order = 0   # Nearest neighbor
    new_shape = (np_image.shape[0], np_image.shape[1]*pixel_spacing[1], np_image.shape[2]*pixel_spacing[2])
    resized_volume = resize(
        np_image,
        output_shape=new_shape,
        order=order,                  # Use trilinear interpolation
        preserve_range=True,      # Crucial: Keep the original data range!
        anti_aliasing=True        # Recommended for downsampling to prevent artifacts
        )
    return resized_volume
def resample(sitk_image, np_image, new_spacing, interpolation_method='nearest', return_sitk_obj=False):
    print(f"start resampling...")
    if interpolation_method =='nearest':
        order = 0   # Nearest neighbor
    elif interpolation_method =='linear':
        order = 1   # Trilinear
    elif interpolation_method =='bspline':
        order = 3   # Cubic B-spline
    else:
        print(f"Warning: Unknown interpolation method '{interpolation_method}'. Defaulting to nearest neighbor.")
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
        
    print(f"finished resampling...")
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
        print("✅ Info: Original PET and CT have opposite Z-axis directions. A final flip will be applied to the NumPy array for correct display.")
    else:
        print("✅ Info: Original PET and CT have the same Z-axis direction. No display flip is needed.")

    # --- (Your existing pre-alignment sanity checks code) ---
    print("--- Before Alignment ---")
    print(f"CT Size: {ct_image.GetSize()}, Origin: {ct_image.GetOrigin()}, Direction: {ct_image.GetDirection()}, Spacing: {ct_image.GetSpacing()}")
    print(f"PET Size: {pet_image.GetSize()}, Origin: {pet_image.GetOrigin()}, Direction: {pet_image.GetDirection()}, Spacing: {pet_image.GetSpacing()}")
    # ... (rest of your verification prints and casting to Float64)

    ct_image = sitk.Cast(ct_image, sitk.sitkFloat64)
    pet_image = sitk.Cast(pet_image, sitk.sitkFloat64)
    
    # ...

    # --- 2. ALIGNMENT (No changes here) ---
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
        print("Applying Z-axis flip to the final array for correct viewing order.")
        resampled_pet_array = resampled_pet_array[::-1, :, :]

    print("\n--- After Alignment ---")
    print(f"Aligned PET Size: {resampled_pet_image.GetSize()}, Origin: {resampled_pet_image.GetOrigin()}, Direction: {resampled_pet_image.GetDirection()}, Spacing: {resampled_pet_image.GetSpacing()}")

    return resampled_pet_image, resampled_pet_array

def transform_to_sitk_image(np_image: np.ndarray, reference_sitk: sitk.Image) -> sitk.Image:
    """
    Converts a NumPy array to a SimpleITK image, copying spatial metadata from a reference image.

    Args:
        np_image (np.ndarray): The input NumPy array to convert.
        reference_sitk (sitk.Image): The reference SimpleITK image to copy metadata from.

    Returns:
        sitk.Image: The resulting SimpleITK image with copied metadata.
    """
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.SetOrigin(reference_sitk.GetOrigin())
    sitk_image.SetSpacing(reference_sitk.GetSpacing())
    sitk_image.SetDirection(reference_sitk.GetDirection())
    return sitk_image


def _permute_and_fix(sitk_image: sitk.Image, perm):
    """Permute axes of a SimpleITK image and fix spacing/direction/origin.

    Args:
        sitk_image: Input 3D SimpleITK image.
        perm: A length-3 list/tuple mapping new axis -> old axis (same semantics as
              SimpleITK.PermuteAxes).

    Returns:
        permuted SimpleITK image with corrected spacing, direction and same origin.
    """
    if len(perm) != 3:
        raise ValueError("perm must be length 3 for a 3D image")

    # Perform the axis permutation
    permuted = sitk.PermuteAxes(sitk_image, perm)

    # Original metadata
    orig_spacing = sitk_image.GetSpacing()  # (sx, sy, sz)
    orig_direction = np.array(sitk_image.GetDirection()).reshape((3, 3))
    orig_origin = sitk_image.GetOrigin()

    # New spacing: pick components according to perm
    new_spacing = tuple(orig_spacing[p] for p in perm)

    # New direction: columns of the direction matrix are permuted
    new_direction_matrix = orig_direction[:, list(perm)]
    new_direction = tuple(new_direction_matrix.flatten())

    permuted.SetSpacing(new_spacing)
    permuted.SetDirection(new_direction)
    permuted.SetOrigin(orig_origin)

    return permuted


def extract_orthogonal_views(sitk_image: sitk.Image):
    """Return axial, coronal and sagittal views as both SimpleITK images and numpy arrays.

    The function permutes axes so that slicing along the last index produces the desired
    anatomical plane. It ensures the resulting images have correct in-plane spacing and
    direction cosines so that visualisation will not be distorted.

    Returns a dict with keys:
      - 'axial': {'sitk': sitk_image, 'np': array, 'spacing': (sx, sy)}
      - 'coronal': {'sitk': coronal_sitk, 'np': array, 'spacing': (sx, sz)}
      - 'sagittal': {'sitk': sagittal_sitk, 'np': array, 'spacing': (sy, sz)}
    """
    # Axial (original) -> in-plane axes are X (left-right) and Y (anterior-posterior)
    axial_sitk = sitk_image
    sx, sy, sz = sitk_image.GetSpacing()
    axial_np = sitk.GetArrayFromImage(axial_sitk)  # shape (z, y, x)

    # Coronal: want in-plane axes X (left-right) and Z (superior-inferior), slice through Y.
    # Permutation mapping new axes -> old axes: [0,2,1] -> (x, z, y)
    coronal_sitk = _permute_and_fix(sitk_image, (0, 2, 1))
    coronal_np = sitk.GetArrayFromImage(coronal_sitk)  # shape (y, z, x) -> slices along last axis -> original Y

    # Sagittal: want in-plane axes Y (anterior-posterior) and Z (superior-inferior), slice through X.
    # Permutation mapping new axes -> old axes: [1,2,0] -> (y, z, x)
    sagittal_sitk = _permute_and_fix(sitk_image, (1, 2, 0))
    sagittal_np = sitk.GetArrayFromImage(sagittal_sitk)  # shape (x, z, y) -> slices along last axis -> original X

    views = {
        'axial': {'sitk': axial_sitk, 'np': axial_np, 'spacing': (sx, sy)},
        'coronal': {'sitk': coronal_sitk, 'np': coronal_np, 'spacing': (sx, sz)},
        'sagittal': {'sitk': sagittal_sitk, 'np': sagittal_np, 'spacing': (sy, sz)},
    }
    return views


def bbox_largest_component(image_array: np.ndarray, mask_array: np.ndarray, display=False):
    """
    Find the largest connected component in a mask and optionally display its bounding box
    over the original image slices.

    Parameters:
        image_array (np.ndarray): 3D image array (z, y, x)
        mask_array (np.ndarray): 3D mask array (same shape as image_array)
        display (bool): Whether to display slices with bounding box overlay

    Returns:
        bbox_slices (tuple): Slices (z_slice, y_slice, x_slice) of the largest component
        bbox_dimensions (tuple): Dimensions (z, y, x) of the bounding box
    """
    # Make mask binary
    binary_mask = mask_array > 0

    # Label connected components
    labeled_mask, num_features = label(binary_mask)
    if num_features == 0:
        raise ValueError("No components found in the mask.")

    # Find slices for each component
    slices = find_objects(labeled_mask)

    # Find largest component
    largest_component_idx = 0
    largest_size = 0
    for i, s in enumerate(slices):
        component = labeled_mask[s] == (i + 1)
        size = np.sum(component)
        if size > largest_size:
            largest_size = size
            largest_component_idx = i

    # Get bounding box slices
    largest_bbox_slices = slices[largest_component_idx]
    z_slice, y_slice, x_slice = largest_bbox_slices

    # Compute bbox dimensions
    bbox_dimensions = (z_slice.stop - z_slice.start,
                       y_slice.stop - y_slice.start,
                       x_slice.stop - x_slice.start)

    if display:
        for z in range(z_slice.start, z_slice.stop):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image_array[z], cmap="gray")
            
            # Draw rectangle (x, y, width, height)
            rect = patches.Rectangle(
                (x_slice.start, y_slice.start),
                bbox_dimensions[2],  # width
                bbox_dimensions[1],  # height
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f"Slice {z}")
            plt.show()

    return largest_bbox_slices, bbox_dimensions

def bbox_full_extent(image_array, mask_array, display=False, cmap = 'gray'):
    """
    Compute the bounding box covering the full extent of a 3D lesion mask.
    
    Parameters:
        image_array (np.ndarray): 3D image array (z, y, x)
        mask_array (np.ndarray): 3D mask array (same shape as image_array)
        display (bool): Whether to display slices with bounding box overlay

    Returns:
        bbox_slices (tuple): Slices (z_slice, y_slice, x_slice) covering all lesion
        bbox_dimensions (tuple): Dimensions (z, y, x) of the bounding box
    """
    # Find the coordinates of all non-zero voxels
    coords = np.argwhere(mask_array > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty.")

    # Get min and max for each dimension
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1  # +1 because slices are exclusive

    # Create slices
    z_slice = slice(z_min, z_max)
    y_slice = slice(y_min, y_max)
    x_slice = slice(x_min, x_max)
    bbox_slices = (z_slice, y_slice, x_slice)

    # Compute dimensions
    bbox_dimensions = (z_max - z_min, y_max - y_min, x_max - x_min)
    box_coordinates = (z_min, y_min, x_min, z_max, y_max, x_max)

    if display:
        for z in range(z_slice.start, z_slice.stop):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image_array[z], cmap=cmap)
            rect = patches.Rectangle(
                (x_slice.start, y_slice.start),
                bbox_dimensions[2],  # width
                bbox_dimensions[1],  # height
                linewidth=2, edgecolor='g', facecolor=(1, 1, 1, 0.2) 
            )
            ax.add_patch(rect)
            ax.set_title(f"Slice {z}")
            plt.show()

    return bbox_slices, bbox_dimensions, box_coordinates

def bbox_per_slice(image_array, mask_array, display=False, cmap='gray'):
    """
    Compute 2D bounding boxes for each slice independently.

    Parameters:
        image_array (np.ndarray): 3D image array (z, y, x)
        mask_array (np.ndarray): 3D mask array (same shape as image_array)
        display (bool): Whether to display slices with bounding box overlay

    Returns:
        bboxes (dict): Dictionary mapping slice index -> 
                       (y_slice, x_slice, (height, width), (y_min, x_min, y_max, x_max))
    """
    if mask_array.shape != image_array.shape:
        raise ValueError("image_array and mask_array must have the same shape.")

    bboxes = {}

    for z in range(mask_array.shape[0]):
        coords = np.argwhere(mask_array[z] > 0)
        if coords.size == 0:
            continue  # skip empty slices

        # Get min and max for y and x
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1  # +1 because slice end is exclusive

        # Build slice objects
        y_slice = slice(y_min, y_max)
        x_slice = slice(x_min, x_max)
        bbox_dimensions = (y_max - y_min, x_max - x_min)
        box_coordinates = (y_min, x_min, y_max, x_max)

        bboxes[z] = (y_slice, x_slice, bbox_dimensions, box_coordinates)

        if display:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image_array[z], cmap=cmap)
            rect = patches.Rectangle(
                (x_min, y_min),
                bbox_dimensions[1],  # width
                bbox_dimensions[0],  # height
                linewidth=1, edgecolor='w', facecolor='black', alpha=0.3
            )
            ax.add_patch(rect)
            ax.set_title(f"Slice {z}")
            plt.show()

    return bboxes

def preprocess_ct(np_image):
    """
    Preprocess CT to remove bed and keep only body (bones + lungs).
    Voxels outside the body are set to the minimum value in the original CT.

    Parameters:
        np_image : np.ndarray
            Input 3D CT volume (Z, H, W)

    Returns:
        final_masked_image : np.ndarray
            CT volume where outside-body voxels = min(original CT)
        final_mask : np.ndarray
            Binary mask of the body (1=body, 0=outside)
    """
    # thresholds to detect bones
    up_t_bones = 3000
    low_t_bones = -200
    mask_bones = np.where((np_image > low_t_bones) & (np_image < up_t_bones), 1, 0)
    mask_lungs = 1 - mask_bones

    mask_lungs_final = np.zeros(mask_lungs.shape, dtype=np.uint8)
    final_mask = np.zeros(mask_lungs.shape, dtype=np.uint8)
    SE = disk(3)
    SE3 = disk(7)

    # process slice by slice
    for i in range(mask_lungs.shape[0]):
        mask_eroded_bones = binary_erosion(mask_bones[i], SE)
        mask_lungs_dilated = binary_dilation(mask_lungs[i], SE)
        mask_cleared_lungs = clear_border(mask_lungs_dilated)
        mask_eroded_lungs = binary_erosion(mask_cleared_lungs, SE)
        mask_filled_bones = binary_fill_holes(mask_eroded_bones)
        mask_filled_lungs = binary_fill_holes(mask_eroded_lungs)
        mask_lungs_connected = remove_small_objects(mask_filled_lungs, min_size=100)
        mask_lungs_final[i] = mask_lungs_connected

        final_mask_temp = mask_filled_bones | mask_lungs_connected
        final_mask_temp = binary_closing(final_mask_temp, SE3)
        final_mask_temp = remove_small_objects(final_mask_temp, min_size=100)
        final_mask[i] = final_mask_temp

    # keep only the largest 3D connected component
    labeled_mask = label(final_mask)
    largest_component = labeled_mask > 0                  # convert to boolean to avoid warning
    largest_component = remove_small_objects(largest_component, min_size=1)
    final_mask_uint8 = largest_component.astype(np.uint8)

    # set voxels outside body to minimum value in original CT
    min_val = np.min(np_image)
    final_masked_image = np.where(largest_component, np_image, min_val)

    return final_masked_image, final_mask_uint8

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

def save_patches_triplets(patches_vol1, patches_vol2, patches_mask, output_dir, cmap_vol1='gray', cmap_vol2='hot', cmap_mask='gray'):
    """
    Save each 2.5D patch as its own image:
    - vol1 triplet (z-1, z, z+1)
    - vol2 triplet (z-1, z, z+1)
    - mask central slice

    Each patch will be saved as 'patch_XXX.png' in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_patches = patches_vol1.shape[0]

    for i in range(n_patches):
        fig, axes = plt.subplots(1, 7, figsize=(14,2))
        
        # vol1 triplet
        for j, dz in enumerate([-1,0,1]):
            axes[j].imshow(patches_vol1[i, dz+1], cmap=cmap_vol1)
            axes[j].axis('off')
            axes[j].set_title(f'vol1 z{dz:+d}')

        # vol2 triplet
        for j, dz in enumerate([-1,0,1]):
            axes[j+3].imshow(patches_vol2[i, dz+1], cmap=cmap_vol2)
            axes[j+3].axis('off')
            axes[j+3].set_title(f'vol2 z{dz:+d}')

        # mask central
        axes[6].imshow(patches_mask[i], cmap=cmap_mask)
        axes[6].axis('off')
        axes[6].set_title('mask center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'patch_{i:03d}.png'))
        plt.close(fig)




def visualize_triplets_full(
    patches_vol1, patches_vol2, target_masks, scribble_masks, prompt_maps,
    save_dir="triplet_visualizations",
    cmap_ct='gray', cmap_pet='hot', cmap_mask='gray',
    prefix="triplet"
):
    """
    Generate and save a figure for each triplet showing:
    CT slices (z-1, z, z+1), PET slices (z-1, z, z+1), 
    Target mask, Scribble mask, Prompt map.
    
    patches_vol1, patches_vol2: (N,3,H,W)
    target_masks, scribble_masks, prompt_maps: (N,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)
    N = patches_vol1.shape[0]

    for i in range(N):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # --- CT triplet slices ---
        for j in range(3):
            axes[0, j].imshow(patches_vol1[i, j], cmap=cmap_ct)
            axes[0, j].set_title(f"CT slice {j-1}")
            axes[0, j].axis('off')

        # --- PET triplet slices ---
        for j in range(3):
            axes[1, j].imshow(patches_vol2[i, j], cmap=cmap_pet)
            axes[1, j].set_title(f"PET slice {j-1}")
            axes[1, j].axis('off')

        # --- Masks / scribbles / prompt ---
        axes[0, 3].imshow(target_masks[i], cmap=cmap_mask)
        axes[0, 3].set_title("Target Mask")
        axes[0, 3].axis('off')

        axes[1, 3].imshow(scribble_masks[i], cmap=cmap_mask)
        axes[1, 3].set_title("Scribble Mask")
        axes[1, 3].axis('off')

        axes[0, 4].imshow(prompt_maps[i], cmap=cmap_pet)
        axes[0, 4].set_title("Prompt Map")
        axes[0, 4].axis('off')

        # Empty subplot for layout alignment
        axes[1, 4].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved figure: {save_path}")

def visualize_propagation_patches(
    patches_vol1, patches_vol2, associated_masks, target_masks,
    save_dir="patch_visualizations",
    cmap_vol1='gray', cmap_vol2='hot', cmap_mask='gray',
    prefix="patch_set"
):
    """
    Generates and saves a figure for each patch set.

    Each figure shows:
    - Volume 1 triplet slices (z-1, z, z+1)
    - Volume 2 triplet slices (z-1, z, z+1)
    - The associated mask (used for centering)
    - The target mask (the ground truth for segmentation)
    """
    os.makedirs(save_dir, exist_ok=True)
    num_patches = len(patches_vol1)
    if num_patches == 0:
        print("No patches to visualize.")
        return

    for i in range(num_patches):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Visualization for Patch Set {i}', fontsize=16)

        # --- Volume 1 triplet slices ---
        for j in range(3):
            ax = axes[0, j]
            im = ax.imshow(patches_vol1[i][j], cmap=cmap_vol1)
            ax.set_title(f"Vol1 slice {j-1}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Volume 2 triplet slices ---
        for j in range(3):
            ax = axes[1, j]
            im = ax.imshow(patches_vol2[i][j], cmap=cmap_vol2)
            ax.set_title(f"Vol2 slice {j-1}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # --- Associated Mask ---
        ax = axes[0, 3]
        im = ax.imshow(associated_masks[i], cmap=cmap_mask)
        ax.set_title("Associated Mask")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Target Mask ---
        ax = axes[1, 3]
        im = ax.imshow(target_masks[i], cmap=cmap_mask)
        ax.set_title("Target Mask")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved figure: {save_path}")

def extract_centered_patch(
    mask: np.ndarray, 
    patch_size: int = 224
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """
    Calculates a fixed-size bounding box and extracts the corresponding patch.

    The function finds the centroid of the mask and defines a square region of
    `patch_size`. If this region exceeds the mask's boundaries, it is shifted
    to fit. Finally, it returns both the coordinates of this region and the
    actual data sliced from the input mask.

    Args:
        mask (np.ndarray): A 2D NumPy array.
        patch_size (int): The desired edge length for the square patch.

    Returns:
        A tuple containing:
            - (tuple): The bounding box `(y_min, y_max, x_min, x_max)`.
            - (np.ndarray): The extracted patch from the mask.
        Returns `None` if the input mask is empty.
    """
    h, w = mask.shape

    # Return None if the mask is empty
    if not mask.any():
        return None
        
    # Calculate the centroid of the non-zero pixels
    cy, cx = center_of_mass(mask)

    # Determine the patch size, capped by the mask's dimensions
    patch_h = min(patch_size, h)
    patch_w = min(patch_size, w)

    # Calculate the top-left corner, shifting it to fit within the boundaries
    y_min = int(round(cy - patch_h / 2))
    x_min = int(round(cx - patch_w / 2))

    if y_min + patch_h > h:
        y_min = h - patch_h
    if x_min + patch_w > w:
        x_min = w - patch_w
    if y_min < 0:
        y_min = 0
    if x_min < 0:
        x_min = 0

    # Define the final bounding box
    bbox = (y_min, y_min + patch_h, x_min, x_min + patch_w)

    # Extract the patch from the mask using the bounding box coordinates
    patch = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    return (bbox, patch)

#OLD VERSION OF SAMPLING FUNCTION (WITH YOUR PROVIDED HELPERS)
# =============================================================================
#  HELPER FUNCTIONS (Your provided functions, with 'netx' corrected to 'nx')
# =============================================================================
def generate_scribble_skeletonization_robust(mask, min_length=20, max_length=60):
    if not np.any(mask): return None
    skeleton = skeletonize(mask)
    skeleton_points = np.argwhere(skeleton)
    if skeleton_points.shape[0] < min_length: return None
    graph = netx.Graph()
    for y, x in skeleton_points:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and skeleton[ny, nx]:
                    graph.add_edge((y, x), (ny, nx))
    path_length = random.randint(min_length, max_length)
    for _ in range(10):
        start_idx, end_idx = np.random.choice(len(skeleton_points), 2, replace=False)
        start, end = tuple(skeleton_points[start_idx]), tuple(skeleton_points[end_idx])
        try:
            path = netx.shortest_path(graph, source=start, target=end)
            if len(path) > path_length: return np.array(path[:path_length])
            elif len(path) >= min_length: return np.array(path)
        except (netx.NetworkXNoPath, netx.NodeNotFound): continue
    start_index = random.randint(0, len(skeleton_points) - min_length)
    return skeleton_points[start_index:start_index + min_length]

def generate_scribble_random_walk(mask, min_length=20, max_length=60):
    valid_points = np.argwhere(mask)
    if valid_points.shape[0] < min_length: return None
    path_length = random.randint(min_length, max_length)
    start_index = random.randint(0, valid_points.shape[0] - 1)
    current_pos = valid_points[start_index]
    scribble = [current_pos]
    for _ in range(path_length - 1):
        found_next = False
        for _ in range(10):
            step = np.random.randint(-1, 2, size=2)
            if np.all(step == 0): continue
            next_pos = current_pos + step
            if (0 <= next_pos[0] < mask.shape[0] and 0 <= next_pos[1] < mask.shape[1] and mask[next_pos[0], next_pos[1]]):
                scribble.append(next_pos)
                current_pos = next_pos
                found_next = True
                break
        if not found_next: break
    if len(scribble) < min_length: return None
    return np.array(scribble)

def generate_prompt_map_with_skimage_dilation(scribble_points, patch_size=(32, 32), sigma=10, dilation_width=1, transformation = 'SDF'):
    scribble_mask = np.zeros(patch_size, dtype=np.uint8)
    if scribble_points is None or len(scribble_points) == 0:
        return np.zeros(patch_size, dtype=np.float32), scribble_mask
    valid = (scribble_points[:, 0] >= 0) & (scribble_points[:, 0] < patch_size[0]) & \
            (scribble_points[:, 1] >= 0) & (scribble_points[:, 1] < patch_size[1])
    scribble_points = scribble_points[valid]
    if len(scribble_points) == 0:
        return np.zeros(patch_size, dtype=np.float32), np.zeros(patch_size, dtype=np.uint8)
    scribble_mask[scribble_points[:, 0], scribble_points[:, 1]] = 1
    if dilation_width > 1:
        radius = dilation_width // 2
        kernel = disk(radius)
        scribble_mask = binary_dilation(scribble_mask, kernel).astype(np.uint8)
    if transformation == 'SDF':
        prompt_map = generate_sdf(scribble_mask, max_dist=patch_size[0]//4)
    elif transformation == 'EXP':
        distance_map = distance_transform_edt(1 - scribble_mask)
        prompt_map = np.exp(-distance_map / sigma)
    return prompt_map, scribble_mask

def generate_prompt_map(mask, sigma = 10):
    distance_map = distance_transform_edt(1 - mask)
    prompt_map = np.exp(-(distance_map**2) / (2 * sigma**2))
    return prompt_map

def generate_sdf(mask, max_dist=None):
    """
    Generate a normalized signed distance map from a binary mask.
    
    mask : numpy array of shape (H, W) or (D, H, W)
           binary mask (1 = object, 0 = background)
    max_dist : float, optional
        Maximum distance for clipping before normalization.
        If None, defaults to 5% of the largest image dimension.
    
    Returns:
        sdf_norm : numpy array in [-1, 1]
    """
    mask = mask.astype(np.uint8)
    shape = mask.shape

    if max_dist is None:
        max_dim = max(shape)
        max_dist = max_dim * 0.05   # 5% of largest dimension

    # distance outside and inside
    dist_out = distance_transform_edt(mask == 0)
    dist_in  = distance_transform_edt(mask == 1)

    # signed distance: positive inside, negative outside
    sdf = dist_in - dist_out

    # clip and normalize
    sdf_clipped = np.clip(sdf, -max_dist, max_dist)
    sdf_norm = sdf_clipped / max_dist  # scale to [-1, 1]

    return sdf_norm.astype(np.float32)

def generate_sdf_complete(mask, label):
    #chek if mask has element greater than 0
    if not np.any(mask): return np.zeros(mask.shape, dtype=np.float32)
    else:
        kernel_size = random.randint(2,8)
        if label == 0: #initialization
            max_dist = random.uniform(mask.shape[0]//5.5, mask.shape[0]//3)
        else: #propagation
            max_dist = random.uniform(mask.shape[0]*0.05, mask.shape[0]//6.5)
        if np.random.rand() < 0.5:
            mask_morph = binary_erosion(mask, disk(kernel_size))
            if not np.any(mask_morph):
                mask_morph = binary_dilation(mask, disk(kernel_size))
        else:
            mask_morph = binary_dilation(mask, disk(kernel_size))
        # distance outside and inside
        dist_out = distance_transform_edt(mask_morph == 0)
        dist_in  = distance_transform_edt(mask_morph == 1)

        # signed distance: positive inside, negative outside
        sdf = dist_in - dist_out

        # clip and normalize
        sdf_clipped = np.clip(sdf, -max_dist, max_dist)
        prompt_map = sdf_clipped / max_dist  # scale to [-1, 1]
        return prompt_map.astype(np.float32)
# =============================================================================
#  MAIN SAMPLING FUNCTION (Final Version with Per-Slice Coverage Rule)
# =============================================================================

def sample_patches_25d_initialization(vol1, vol2, mask, patch_size=(32, 32),
                                scribble_min_length=20, scribble_max_length=60,
                                scribble_dilation=1, prompt_sigma=10, min_coverage=0.3, transformation = 'SDF'):
    Z, H, W = vol1.shape
    ph, pw = patch_size

    z_idx = np.any(mask, axis=(1, 2))
    if not np.any(z_idx):
        raise ValueError("No slices with mask!")

    zmin, zmax = np.where(z_idx)[0][[0, -1]]

    patches_vol1, patches_vol2, patches_mask = [], [], []
    patches_scribble_mask, patches_prompt_map = [], []

    def get_triplet(vol, z, y, x):
        triplet = []
        for dz in [-1, 0, 1]:
            zz = np.clip(z + dz, 0, Z - 1)
            triplet.append(vol[zz, y:y + ph, x:x + pw])
        return np.stack(triplet, axis=0)

    # --- Unified Sampling Logic ---
    all_coords_to_sample = []
    
    # 1. Get coordinates for POSITIVE samples
    target_slices = np.where(z_idx)[0]
    for z in target_slices:
        coords = np.argwhere(mask[z] > 0)
        cy, cx = coords.mean(axis=0).astype(int)
        all_coords_to_sample.append({'z': z, 'cy': cy, 'cx': cx, 'type': 'positive'})
    
    N_target = len(all_coords_to_sample)
    if N_target == 0: raise ValueError("Could not find any positive centers!")

    # 2. Get coordinates for NEGATIVE samples
    n_nonmask = N_target
    n70 = int(0.7 * n_nonmask)
    n10 = int(0.1 * n_nonmask)
    n20 = n_nonmask - (n70 + n10)

    vals_target = vol2[mask > 0]
    thr = vals_target.min() if len(vals_target) > 0 else vol2.mean()
    
    eligible_negative_coords = np.argwhere((mask == 0) & (vol2 >= thr))
    
    np.random.shuffle(eligible_negative_coords)
    z_coords = eligible_negative_coords[:, 0]
    
    z_target_area = np.arange(max(0, zmin - 1), min(Z, zmax + 2))
    coords_70 = eligible_negative_coords[np.isin(z_coords, z_target_area)][:n70]
    coords_10 = eligible_negative_coords[z_coords < zmin][:n10]
    coords_20 = eligible_negative_coords[z_coords > zmax][:n20]

    for z, cy, cx in np.vstack([coords_70, coords_10, coords_20]):
        all_coords_to_sample.append({'z': z, 'cy': cy, 'cx': cx, 'type': 'negative'})

    # --- Process all selected coordinates ---
    for sample in all_coords_to_sample:
        scribble_pts = None
        z, cy, cx = sample['z'], sample['cy'], sample['cx']
        
        y = np.clip(cy - ph // 2, 0, H - ph)
        x = np.clip(cx - pw // 2, 0, W - pw)

        patch_mask = mask[z, y:y + ph, x:x + pw]
        patch_vol2 = get_triplet(vol2, z, y, x)
        
        # --- NEW: Per-Slice Coverage Calculation ---
        total_slice_mask_voxels = np.sum(mask[z])
        # Avoid division by zero if the slice is unexpectedly empty
        if total_slice_mask_voxels > 0:
            slice_coverage_threshold = min_coverage * total_slice_mask_voxels
        else:
            slice_coverage_threshold = 0
            
        is_positive_sample = (sample['type'] == 'positive') or (
        np.sum(patch_mask) > 0 and np.sum(patch_mask) >= slice_coverage_threshold)
        # --- END NEW ---
        
        final_patch_mask = patch_mask if is_positive_sample else np.zeros_like(patch_mask)

        # --- Scribble Generation with Fallback ---

        if is_positive_sample:
            if np.random.rand() < 0.5:
                scribble_pts =  generate_scribble_skeletonization_robust(final_patch_mask, scribble_min_length, scribble_max_length)
            else:
                scribble_pts = generate_scribble_random_walk(final_patch_mask, scribble_min_length, scribble_max_length)

            if scribble_pts is None or len(scribble_pts) == 0:
                print(f" skeletonization and random walk failed")
                coords = np.argwhere(final_patch_mask > 0)
                if len(coords) > 0:
                    idxs = np.random.choice(len(coords), 
                                    size=min(len(coords), scribble_max_length), 
                                    replace=False)
                    scribble_pts = coords[idxs]

            if scribble_pts is None or len(scribble_pts) == 0:
                print(f" positive scribbling failed")


        else:
            pet_center_slice = patch_vol2[1]
            pet_thr = thr * 0.7
            pseudo_mask_from_pet = (pet_center_slice >= pet_thr)
            if np.random.rand() < 0.5:
                scribble_pts =  generate_scribble_skeletonization_robust(pseudo_mask_from_pet, scribble_min_length, scribble_max_length)
            else:
                scribble_pts = generate_scribble_random_walk(pseudo_mask_from_pet, scribble_min_length, scribble_max_length)

            if scribble_pts is None or len(scribble_pts) == 0:
                print(f"skeletonization and random walk failed")
                coords = np.argwhere(pseudo_mask_from_pet > 0)
                if len(coords) > 0:
                    idxs = np.random.choice(len(coords), 
                                    size=min(len(coords), scribble_max_length), 
                                    replace=False)
                    scribble_pts = coords[idxs]
            
            if scribble_pts is None or len(scribble_pts) == 0:
                print(f" negative scribbling failed")
                
        if scribble_pts is None:
            continue
            
        prompt_map, scribble_mask_patch = generate_prompt_map_with_skimage_dilation(
            scribble_pts, patch_size=(ph, pw), sigma=prompt_sigma, dilation_width=scribble_dilation, transformation = transformation)

        patches_vol1.append(get_triplet(vol1, z, y, x))
        patches_vol2.append(patch_vol2)
        patches_mask.append(final_patch_mask)
        patches_scribble_mask.append(scribble_mask_patch)
        patches_prompt_map.append(prompt_map)

    return (np.array(patches_vol1), np.array(patches_vol2), np.array(patches_mask),
            np.array(patches_scribble_mask))


import numpy as np

def sample_patches_25d_breast(vol1, vol2, mask, view='axial', patch_size=(32, 32), 
                              min_coverage=0.3, contra_share=0.85):
    """
    Final Optimized Sampler with View-Dependent Ratios.
    
    Ratios (Automatic based on view):
    - Axial:    50% Positive / 50% Negative (1.0 ratio)
    - Coronal:  70% Positive / 30% Negative (~0.43 ratio)
    - Sagittal: 70% Positive / 30% Negative (~0.43 ratio)
    
    Args:
        contra_share (float): Of the generated negatives, what % should be Contralateral?
                              (Default 0.85 -> 85% Contralateral, 15% Random Background)
    """
    # 1. Float32 Enforcement (Crucial for Memory)
    if vol1.dtype == np.float64: vol1 = vol1.astype(np.float32)
    if vol2.dtype == np.float64: vol2 = vol2.astype(np.float32)

    Z, H, W = vol1.shape
    ph, pw = patch_size
    ph_half, pw_half = ph // 2, pw // 2

    # Helper: Extract 3-slice triplet
    def get_triplet(vol, z, y, x):
        triplet = []
        for dz in [-1, 0, 1]:
            zz = np.clip(z + dz, 0, Z - 1)
            triplet.append(vol[zz, y:y + ph, x:x + pw])
        return np.stack(triplet, axis=0)

    # --- 2. Determine Ratios based on View ---
    view_lower = view.lower()
    
    if view_lower == 'axial':
        # 50/50 Split -> 1 Negative for every 1 Positive
        neg_to_pos_ratio = 1.0 
    else: 
        # 70% Pos / 30% Neg -> 0.428 Negatives for every 1 Positive
        # Math: 30 / 70 = 0.42857...
        neg_to_pos_ratio = 30 / 70

    # --- 3. Get POSITIVE Samples (Centroid per slice) ---
    final_positives = []
    
    # Identify slices with mask
    z_indices = np.where(np.any(mask, axis=(1, 2)))[0]
    if len(z_indices) == 0: raise ValueError(f"No slices with mask in {view} view!")

    slice_mask_sums = np.sum(mask, axis=(1, 2))

    for z in z_indices:
        ys, xs = np.where(mask[z] > 0)
        if len(ys) == 0: continue
        
        # Centroid
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        y = np.clip(cy - ph_half, 0, H - ph)
        x = np.clip(cx - pw_half, 0, W - pw)
        
        final_positives.append((z, y, x))

    N_pos = len(final_positives)
    if N_pos == 0: raise ValueError("No positive patches found.")

    # --- 4. Calculate Negative Targets ---
    total_negatives_needed = int(np.ceil(N_pos * neg_to_pos_ratio))
    
    # Split Negatives: Mostly Contralateral (if available), rest Threshold
    target_contra = int(np.ceil(total_negatives_needed * contra_share))
    target_thresh = total_negatives_needed - target_contra

    candidate_negatives_contra = []

    # --- 5. Generate Contralateral (With Promotion) ---
    # Only applicable for Axial/Coronal usually, but logic holds if symmetric
    if view_lower in ['axial', 'coronal']:
        current_positives = list(final_positives)
        
        for (z, y, x) in current_positives:
            # Flip X
            center_x = x + pw_half
            flipped_center_x = W - 1 - center_x
            x_contra = int(np.clip(flipped_center_x - pw_half, 0, W - pw))
            
            # Check overlap
            patch_mask_sum = np.sum(mask[z, y:y+ph, x_contra:x_contra+pw])
            slice_total = slice_mask_sums[z]
            thresh = min_coverage * slice_total if slice_total > 0 else 0
            
            if patch_mask_sum == 0:
                candidate_negatives_contra.append((z, y, x_contra))
            elif patch_mask_sum >= thresh:
                # Promote to Positive (if not already there)
                if (z, y, x_contra) not in final_positives:
                    final_positives.append((z, y, x_contra))

    # Shuffle and Select Contralateral
    np.random.shuffle(candidate_negatives_contra)
    
    if len(candidate_negatives_contra) < target_contra:
        selected_contra = candidate_negatives_contra
        missing = target_contra - len(candidate_negatives_contra)
        target_thresh += missing # Fill gap with random background
    else:
        selected_contra = candidate_negatives_contra[:target_contra]

    # --- 6. Generate Threshold Negatives (Rejection Sampling) ---
    selected_thresh = []
    
    if target_thresh > 0:
        mask_bool = mask > 0
        thr = vol2[mask_bool].min() if np.any(mask_bool) else vol2.mean()
        
        # Distribution: 70% near tumor z-slices, 10% above, 20% below
        n70 = int(0.7 * target_thresh)
        n10 = int(0.1 * target_thresh)
        n20 = target_thresh - n70 - n10
        
        zmin, zmax = z_indices[0], z_indices[-1]

        def get_random_negatives(z_start, z_end, count):
            if count <= 0 or z_end <= z_start: return []
            found = []
            attempts = 0
            max_attempts = count * 200
            
            while len(found) < count and attempts < max_attempts:
                attempts += 1
                rz = np.random.randint(z_start, z_end)
                ry = np.random.randint(0, H - ph)
                rx = np.random.randint(0, W - pw)
                
                if mask[rz, ry+ph_half, rx+pw_half] == 0: # Center Check
                    if vol2[rz, ry+ph_half, rx+pw_half] >= thr: # Intensity Check
                        if np.sum(mask[rz, ry:ry+ph, rx:rx+pw]) == 0: # Full Patch Check
                            found.append((rz, ry, rx))
            return found

        selected_thresh.extend(get_random_negatives(max(0, zmin-1), min(Z, zmax+2), n70))
        selected_thresh.extend(get_random_negatives(0, zmin, n10))
        selected_thresh.extend(get_random_negatives(zmax, Z, n20))

    # --- 7. Final Extraction ---
    # Combine lists with labels: 1=Pos, 0=Neg
    all_coords = []
    for c in final_positives: all_coords.append(c + (1,))
    for c in selected_contra: all_coords.append(c + (0,))
    for c in selected_thresh: all_coords.append(c + (0,))
    
    total_patches = len(all_coords)
    
    # Allocating memory (Small size: ~100-200 patches)
    out_vol1 = np.zeros((total_patches, 3, ph, pw), dtype=np.float32)
    out_vol2 = np.zeros((total_patches, 3, ph, pw), dtype=np.float32)
    out_mask = np.zeros((total_patches, ph, pw), dtype=mask.dtype)

    for i, (z, y, x, is_pos) in enumerate(all_coords):
        out_vol1[i] = get_triplet(vol1, z, y, x)
        out_vol2[i] = get_triplet(vol2, z, y, x)
        if is_pos:
            out_mask[i] = mask[z, y:y+ph, x:x+pw]

    return out_vol1, out_vol2, out_mask

import numpy as np

def sample_patches_25d_prostate(vol1, vol2, mask, view='axial', patch_size=(32, 32), min_coverage=0.3):
    """
    Prostate-Specific Sampling (2.5D) with Promotion.
    
    Logic:
    1. POSITIVES: 1 Centroid patch per slice containing mask.
    2. NEGATIVES: Threshold-based random background sampling only.
       - Promotion: If a 'background' candidate overlaps mask >= 30%, it becomes POSITIVE.
    3. RATIO: Fixed at 70% Positive / 30% Negative.
    """
    # 1. Float32 Enforcement
    if vol1.dtype == np.float64: vol1 = vol1.astype(np.float32)
    if vol2.dtype == np.float64: vol2 = vol2.astype(np.float32)

    Z, H, W = vol1.shape
    ph, pw = patch_size
    ph_half, pw_half = ph // 2, pw // 2

    # Helper: Extract 3-slice triplet
    def get_triplet(vol, z, y, x):
        triplet = []
        for dz in [-1, 0, 1]:
            zz = np.clip(z + dz, 0, Z - 1)
            triplet.append(vol[zz, y:y + ph, x:x + pw])
        return np.stack(triplet, axis=0)

    # --- 2. Get Initial POSITIVE Samples (Centroid per slice) ---
    final_positives = []
    
    # Pre-calculate slice sums for promotion check
    slice_mask_sums = np.sum(mask, axis=(1, 2))
    z_indices = np.where(slice_mask_sums > 0)[0]
    
    if len(z_indices) == 0: 
        raise ValueError(f"No positive mask found in {view} view!")

    for z in z_indices:
        ys, xs = np.where(mask[z] > 0)
        if len(ys) == 0: continue
        
        # Centroid
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        y = np.clip(cy - ph_half, 0, H - ph)
        x = np.clip(cx - pw_half, 0, W - pw)
        
        final_positives.append((z, y, x))

    # Create a set for fast duplicate checking during promotion
    existing_pos_set = set(final_positives)

    N_pos = len(final_positives)
    if N_pos == 0: raise ValueError("No positive patches found.")

    # --- 3. Calculate Negative Targets (70/30 Fixed Ratio) ---
    # Ratio: 30 negatives for every 70 positives
    neg_to_pos_ratio = 30 / 70
    target_thresh = int(np.ceil(N_pos * neg_to_pos_ratio))

    # --- 4. Generate Threshold Negatives (With Promotion) ---
    selected_thresh = []
    
    if target_thresh > 0:
        mask_bool = mask > 0
        thr = vol2[mask_bool].min() if np.any(mask_bool) else vol2.mean()
        
        # Spatial Splits
        n70 = int(0.7 * target_thresh)
        n10 = int(0.1 * target_thresh)
        n20 = target_thresh - n70 - n10
        
        zmin, zmax = z_indices[0], z_indices[-1]

        def get_candidates_with_promotion(z_start, z_end, count):
            if count <= 0 or z_end <= z_start: return []
            
            found_negatives = []
            attempts = 0
            max_attempts = count * 200 # Safety break
            
            while len(found_negatives) < count and attempts < max_attempts:
                attempts += 1
                rz = np.random.randint(z_start, z_end)
                ry = np.random.randint(0, H - ph)
                rx = np.random.randint(0, W - pw)
                
                # 1. Center Check (Must look like background initially)
                if mask[rz, ry+ph_half, rx+pw_half] == 0:
                    
                    # 2. Intensity Check
                    if vol2[rz, ry+ph_half, rx+pw_half] >= thr:
                        
                        # 3. Promotion Logic
                        patch_mask_sum = np.sum(mask[rz, ry:ry+ph, rx:rx+pw])
                        
                        if patch_mask_sum == 0:
                            # Strict Negative (No overlap)
                            found_negatives.append((rz, ry, rx))
                        else:
                            # Check for Promotion
                            slice_total = slice_mask_sums[rz]
                            coverage_thresh = min_coverage * slice_total if slice_total > 0 else 0
                            
                            if patch_mask_sum >= coverage_thresh:
                                # Promote to Positive
                                if (rz, ry, rx) not in existing_pos_set:
                                    final_positives.append((rz, ry, rx))
                                    existing_pos_set.add((rz, ry, rx))
                            
                            # If 0 < sum < coverage_thresh, we discard (Ambiguous)
            
            return found_negatives

        # Gather samples
        selected_thresh.extend(get_candidates_with_promotion(max(0, zmin-1), min(Z, zmax+2), n70))
        selected_thresh.extend(get_candidates_with_promotion(0, zmin, n10))
        selected_thresh.extend(get_candidates_with_promotion(zmax, Z, n20))

    # --- 5. Final Extraction ---
    # Combine coordinates: Positives (Label 1) and Negatives (Label 0)
    all_coords = []
    for c in final_positives: all_coords.append(c + (1,))
    for c in selected_thresh: all_coords.append(c + (0,))
    
    total_patches = len(all_coords)
    
    # Pre-allocate memory (Float32)
    out_vol1 = np.zeros((total_patches, 3, ph, pw), dtype=np.float32)
    out_vol2 = np.zeros((total_patches, 3, ph, pw), dtype=np.float32)
    out_mask = np.zeros((total_patches, ph, pw), dtype=mask.dtype)

    for i, (z, y, x, is_pos) in enumerate(all_coords):
        out_vol1[i] = get_triplet(vol1, z, y, x)
        out_vol2[i] = get_triplet(vol2, z, y, x)
        
        if is_pos:
            out_mask[i] = mask[z, y:y+ph, x:x+pw]
        # Negatives remain 0

    return out_vol1, out_vol2, out_mask


def sample_patches_2d_propagation(vol1, vol2, mask, patch_size=(32, 32)):
    """
    Extracts 2.5D patches from two volumes based on a mask for segmentation tasks.

    The process is as follows:
    1. For each slice 'z' containing a mask, a 3-slice triplet centered at z+1 is
       extracted. This triplet is spatially centered on the centroid of the mask at 'z'.
    2. A single additional triplet is extracted, centered at z_first-1, where
       z_first is the first slice containing a mask. This triplet is also centered
       on the centroid of the mask at z_first.

    Args:
        vol1 (np.ndarray): The first input volume (Z, H, W).
        vol2 (np.ndarray): The second input volume (Z, H, W).
        mask (np.ndarray): The mask volume (Z, H, W).
        patch_size (tuple): The height and width of the patches (ph, pw).

    Returns:
        tuple: A tuple containing four lists:
               - patches_vol1 (list): 3xPHxPW patches from vol1.
               - patches_vol2 (list): 3xPHxPW patches from vol2.
               - patches_mask_associated (list): PHxPW patches from the original mask
                 slice ('z') used for centering.
               - patches_mask_target (list): PHxPW patches from the triplet's center
                 slice, representing the segmentation target.
    """
    if not (vol1.shape == vol2.shape == mask.shape):
        raise ValueError("All input volumes (vol1, vol2, mask) must have the same shape.")

    Z, H, W = vol1.shape
    ph, pw = patch_size
    
    z_idx = np.any(mask, axis=(1, 2))
    target_slices = np.where(z_idx)[0]

    if target_slices.size == 0:
        print("Warning: No slices with a mask were found. Returning empty lists.")
        return [], [], [], []

    patches_vol1, patches_vol2, patches_mask_associated, patches_mask_target = [], [], [], []

    def _get_and_append_patches(center_z, mask_z):
        """Helper function to extract and append a set of patches."""
        mask_slice = mask[mask_z]
        try:
            cy, cx = map(int, center_of_mass(mask_slice))
        except ValueError:
            coords = np.argwhere(mask_slice > 0)
            if coords.size == 0: return
            cy, cx = coords.mean(axis=0).astype(int)

        y_start = max(0, cy - ph // 2)
        x_start = max(0, cx - pw // 2)
        y_start = min(y_start, H - ph)
        x_start = min(x_start, W - pw)
        y_end, x_end = y_start + ph, x_start + pw

        # Associated Mask Patch (from the slice that provides the center)
        associated_mask_patch = mask[mask_z, y_start:y_end, x_start:x_end]
        
        # Target Mask Patch (from the triplet's center slice)
        target_z_clipped = np.clip(center_z, 0, Z - 1)
        target_mask_patch = mask[target_z_clipped, y_start:y_end, x_start:x_end]

        # Volume Triplets (centered around center_z)
        vol1_triplet = np.zeros((3, ph, pw), dtype=vol1.dtype)
        vol2_triplet = np.zeros((3, ph, pw), dtype=vol2.dtype)

        for i, dz in enumerate([-1, 0, 1]):
            zz = np.clip(center_z + dz, 0, Z - 1)
            vol1_triplet[i, :, :] = vol1[zz, y_start:y_end, x_start:x_end]
            vol2_triplet[i, :, :] = vol2[zz, y_start:y_end, x_start:x_end]
        
        patches_vol1.append(vol1_triplet)
        patches_vol2.append(vol2_triplet)
        patches_mask_associated.append(associated_mask_patch)
        patches_mask_target.append(target_mask_patch)

    # --- 1. The single additional triplet for the initial boundary ---
    # Centered at z_first-1, associated with mask at z_first.
    z_first = target_slices[0]
    _get_and_append_patches(center_z=z_first - 1, mask_z=z_first)

    # --- 2. The main set of triplets for propagation ---
    # For each mask slice 'z', get the triplet centered at 'z+1'.
    for z in target_slices:
        _get_and_append_patches(center_z=z + 1, mask_z=z)

    return np.array(patches_vol1), np.array(patches_vol2), np.array(patches_mask_associated), np.array(patches_mask_target)

def compute_mask_percentage(patches):
    # Compute the sum of ones per patch (mask volume in that patch)
    patch_sums = np.array([np.sum(patch) for patch in patches])
    # Compute percentages relative to total mask volume
    total_mask_volume = np.sum(patch_sums)
    if total_mask_volume > 0:
        percentages = patch_sums / total_mask_volume
    else:
        percentages = np.zeros_like(patch_sums, dtype=float)
    
    # Min-max normalization
    if len(percentages) > 0 and np.max(percentages) != np.min(percentages):
        weights = (percentages - np.min(percentages)) / (np.max(percentages) - np.min(percentages))
    else:
        weights = np.zeros_like(percentages, dtype=float)

    return weights

def save_multimodal_volumes_slices(multimodal_volume, target_mask, output_dir):
    for i in range(multimodal_volume.shape[0]):
        np.savez_compressed(
            os.path.join(output_dir, f"multimodal_volume_slice_{i:03d}.npz"),
            volume=multimodal_volume[i,:,:,:]
        )
        np.savez_compressed(
            os.path.join(output_dir, f"target_mask_slice_{i:03d}.npz"),
            mask=target_mask[i,:,:])

def save_multimodal_volumes_slices_2(multimodal_volume, target_mask, view, organ, output_dir):
    for i in range(multimodal_volume.shape[0]):
        np.savez_compressed(
            os.path.join(output_dir, f"multimodal_volume_{organ}_{view}_slice_{i:03d}.npz"),
            volume=multimodal_volume[i,:,:,:]
        )
        np.savez_compressed(
            os.path.join(output_dir, f"target_mask_{organ}_{view}_slice_{i:03d}.npz"),
            mask=target_mask[i,:,:])

def visualize_multimodal_triplets(
    multimodal_volume, target_masks, labels, weights,
    save_dir="triplet_visualizations",
    cmap_ct='gray', cmap_pet='hot', cmap_mask='gray',
    prefix="triplet"
):
    """
    Generate and save a figure for each sample showing:
    CT slices (z-1, z, z+1), PET slices (z-1, z, z+1), 
    Target mask, Scribble mask, Prompt map.

    multimodal_volume: (N, 7, H, W)
        channels 0-2: CT triplet
        channels 3-5: PET triplet
        channel 6: scribble mask
    target_masks, prompt_maps: (N,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)
    N = multimodal_volume.shape[0]

    for i in range(N):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # --- CT triplet slices ---
        for j in range(3):
            axes[0, j].imshow(multimodal_volume[i, j], cmap=cmap_ct)
            axes[0, j].set_title(f"CT slice {j-1}")
            axes[0, j].axis('off')

        # --- PET triplet slices ---
        for j in range(3):
            axes[1, j].imshow(multimodal_volume[i, j+3], cmap=cmap_pet)
            axes[1, j].set_title(f"PET slice {j-1}")
            axes[1, j].axis('off')

        # --- Masks / scribbles / prompt ---
        axes[0, 3].imshow(target_masks[i], cmap=cmap_mask)
        axes[0, 3].set_title("Target Mask")
        axes[0, 3].axis('off')

        axes[1, 3].imshow(multimodal_volume[i, 6], cmap=cmap_mask)  # scribble mask channel
        axes[1, 3].set_title("Scribble Mask" + f" (label={labels[i]}), (w={weights[i]:.2f})")
        axes[1, 3].axis('off')

        prompt_map = generate_sdf_complete(multimodal_volume[i, 6], labels[i])
        axes[0, 4].imshow(prompt_map, cmap=cmap_pet)
        axes[0, 4].set_title("Prompt Map" + f" (label={labels[i]}), (w={weights[i]:.2f})")
        axes[0, 4].axis('off')

        # Empty subplot for layout alignment
        axes[1, 4].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved figure: {save_path}")


def visualize_multimodal_triplets_2(
    multimodal_volume, target_masks,
    save_dir="triplet_visualizations_2",
    cmap_ct='gray', cmap_pet='hot', cmap_mask='gray',
    view = 'axial',
    prefix="triplet2"
):
    """
    Visualize multimodal volumes with shape (N,6,H,W):
      channels 0-2: CT triplet
      channels 3-5: PET triplet

    Shows CT triplet (top row), PET triplet (bottom row) and target mask (top-right).
    """
    os.makedirs(save_dir, exist_ok=True)
    N = multimodal_volume.shape[0]

    for i in range(N):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        # CT triplet (channels 0-2)
        for j in range(3):
            axes[0, j].imshow(multimodal_volume[i, j], cmap=cmap_ct)
            axes[0, j].set_title(f"CT slice {j-1}")
            axes[0, j].axis('off')

        # PET triplet (channels 3-5)
        for j in range(3):
            axes[1, j].imshow(multimodal_volume[i, j+3], cmap=cmap_pet)
            axes[1, j].set_title(f"PET slice {j-1}")
            axes[1, j].axis('off')

        # Target mask
        axes[0, 3].imshow(target_masks[i], cmap=cmap_mask)
        axes[0, 3].set_title("Target Mask")
        axes[0, 3].axis('off')

        # Empty subplot for layout alignment
        axes[1, 3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{view}_{prefix}_{i:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved figure: {save_path}")

def get_centroid(mask):
    """Calculates center of mass (z, y, x) of the binary mask."""
    if not np.any(mask):
        return mask.shape[0]//2, mask.shape[1]//2, mask.shape[2]//2
    coords = center_of_mass(mask > 0)
    return int(round(coords[0])), int(round(coords[1])), int(round(coords[2]))

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
    # Note: The user's formula implies distance from the scribbles.
    # distance_transform_edt(input) calculates distance to nearest BACKGROUND (0).
    # So we calculate distance on (1 - scribble).
    distance_map = distance_transform_edt(1 - scribble_binary)
    
    prompt_map = np.exp(-distance_map / sigma)
    return prompt_map.astype(np.float32)

def reconstruct_and_resample_back_optimized(pred_patch_2d, metadata, original_sitk_reference):
    """
    Step 3: RECONSTRUCTION & RESAMPLING BACK
    1. Places the 224x224 prediction patch into the empty Resampled Volume.
    2. Assigns spatial metadata (Spacing 1.17, Origin, etc.) to that volume.
    3. Resamples that volume back to the Original CT Geometry (Viewer Space).
    """
    import SimpleITK as sitk
    import numpy as np

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
    seg_sitk.CopyInformation(sitk_resampled_ref) # CRITICAL: Copy info from Resampled CT

    # 4. Resample BACK to Viewer Space (Original CT Geometry)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_sitk_reference) # Target: The original CT user is viewing
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest Neighbor for Binary Masks
    resampler.SetDefaultPixelValue(0)
    
    # The transform is Identity because the spatial coordinates (Origin/Direction) 
    # handles the alignment automatically.
    final_seg_sitk = resampler.Execute(seg_sitk)
    
    # 5. Return Numpy Array (matches self.segmentation_mask shape)
    final_seg_numpy = sitk.GetArrayFromImage(final_seg_sitk)
    
    return final_seg_numpy

def run_segmentation_pipeline(sitk_ct, numpy_ct, sitk_pet_aligned, numpy_pet_aligned, sitk_mask, numpy_mask):
    """
    Runs pipeline and returns patches PLUS spatial metadata needed for reconstruction.
    """
    print("--- Starting Pipeline ---")
    target_spacing = (1.171875, 1.171875, 3.0)

    # 1. Resample
    print("Resampling Volumes...")
    # We need the SimpleITK object of the resampled CT to serve as a reference later
    sitk_ct_resampled = resample(sitk_ct, numpy_ct, target_spacing, 'linear', return_sitk_obj=True)
    resampled_np_ct = sitk.GetArrayFromImage(sitk_ct_resampled)
    
    # We don't strictly need the intermediate SITK objects for PET/Mask, just arrays
    resampled_np_pet = resample(sitk_pet_aligned, numpy_pet_aligned, target_spacing, 'linear')
    resampled_np_mask = resample(sitk_mask, numpy_mask, target_spacing, 'nearest')

    # 2. Preprocess
    vol_ct_final = apply_window(resampled_np_ct, center=40, width=400)
    vol_pet_final = min_max_normalize(resampled_np_pet)
    
    # 3. Extract Patches
    print("Extracting Centered Patches...")
    cz, cy, cx = get_centroid(resampled_np_mask)
    
    patch_ct = extract_25d_patch_centered(vol_ct_final, cz, cy, cx, 224)
    patch_pet = extract_25d_patch_centered(vol_pet_final, cz, cy, cx, 224)
    patch_mask = extract_25d_patch_centered(resampled_np_mask, cz, cy, cx, 224)

    # 4. Prompt Map
    center_mask_slice = patch_mask[1] # Use central slice
    prompt_map_2d = generate_prompt_map(center_mask_slice, sigma=20)
    prompt_map_channel = np.expand_dims(prompt_map_2d, axis=0)

    # 5. Stack
    final_input_volume = np.concatenate([patch_ct, patch_pet, prompt_map_channel], axis=0)

    # --- RETURN DATA PLUS METADATA ---
    metadata = {
        'centroid': (cz, cy, cx),       # Where the patch belongs
        'shape': resampled_np_ct.shape, # The size of the full resampled volume
        'sitk_ref': sitk_ct_resampled   # The spatial reference for inverse resampling
    }

    return final_input_volume, metadata

def prepare_volumes_for_inference(sitk_ct, numpy_ct, sitk_pet_aligned, numpy_pet_aligned):
    """
    STEP 1: HEAVY PREPROCESSING (Run ONCE on Load)
    - Resamples volumes to target spacing (1.17, 1.17, 3.0).
    - Windows CT and Normalizes PET.
    - Stores the 'Segmentation Objects' in memory.
    """
    print("\n--- Pipeline: Initialization (Resampling & Normalizing) ---")
    target_spacing = (1.171875, 1.171875, 3.0)

    # 1. Resample CT (Keep SITK object for reference)
    # We use your existing 'resample' function
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

    print("Pipeline: Initialization Done.")
    return vol_ct_final, vol_pet_final, scale_info


def run_inference_on_preprocessed(vol_ct_final, vol_pet_final, scale_info, mask_input_viewer_2d, viewer_slice_idx):
    """
    STEP 2: FAST INFERENCE (Run on 'Start Segmentation')
    - Maps user click/scribble to Resampled Space.
    - Extracts patches using your specific 'extract_25d_patch_centered'.
    - Generates prompt map using your specific 'generate_prompt_map'.
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

    # 3. Calculate Centroid (using your logic)
    # We construct a temporary 3D mask just to use your get_centroid function, 
    # or just calculate 2D centroid here. Let's use 2D calculation for speed.
    if np.sum(mask_resampled_2d) > 0:
        coords = center_of_mass(mask_resampled_2d > 0)
        cy, cx = int(round(coords[0])), int(round(coords[1]))
    else:
        cy, cx = resampled_h // 2, resampled_w // 2
    
    cz = z_resampled # The Z is determined by the slice we are on

    # 4. Extract Patches (Using YOUR function)
    patch_ct = extract_25d_patch_centered(vol_ct_final, cz, cy, cx, 224)
    patch_pet = extract_25d_patch_centered(vol_pet_final, cz, cy, cx, 224)
    
    # Extract Mask Patch (locally)
    # We need to create a small 3D volume for the mask to use your extractor
    # or manually crop the 2D mask. 
    # To keep strictly to your logic:
    # We create a dummy 3D mask volume (mostly zeros) just for this slice? 
    # NO, that's too slow. We manually crop the 2D mask using the logic inside extract_25d_patch_centered.
    
    # Manual Crop of 2D mask (Logic copied from extract_25d_patch_centered)
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

    # 5. Prompt Map (Using YOUR function)
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

# ==========================================
# PART 4: VOLUMETRIC PROPAGATION TOOLS
# ==========================================

def run_inference_direct_ai_space(vol_ct_final, vol_pet_final, mask_input_2d, z_ai):
    """
    Inference helper for the propagation loop.
    Unlike the main inference function, this works PURELY in AI-Space (Resampled coordinates).
    It does not perform coordinate mapping from the viewer.
    """
    import numpy as np
    from scipy.ndimage import center_of_mass

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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    import SimpleITK as sitk
    
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

# ==========================================
# PART 5: GEOMETRIC ROBUSTNESS TOOLS
# ==========================================

def check_overlap_consistency(prev_mask, curr_mask):
    """
    ROBUSTNESS: Overlap Constraint (IoU check).
    Ensures the segmentation doesn't 'jump' to a completely new area 
    disconnected from the previous slice.
    """
    intersection = np.logical_and(prev_mask, curr_mask)
    print(f"intersection {np.max(intersection)}")
    union = np.logical_or(prev_mask, curr_mask)
    print(f"union: {np.max(union)}")
    
    # If both are empty, that's consistent (end of tumor)
    if np.sum(union) == 0: 
        return True
    
    # If previous mask existed, but there is NO intersection with new mask,
    # it means the model hallucinated a shape somewhere else. Stop.
    if np.sum(prev_mask) > 0 and np.sum(intersection) == 0:
        return False
        
    return True

def keep_largest_connected_component_3d(vol_mask):
    """
    ROBUSTNESS: 3D Connectivity.
    Removes floating artifacts/islands. Keeps only the largest connected volume.
    """
    from skimage.measure import label
    import numpy as np
    
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
    import numpy as np
    
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