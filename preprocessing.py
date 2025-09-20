import os
import SimpleITK as sitk
import numpy as np
from pydicom import dcmread
import pydicom

# Allow pydicom to convert wrong-length elements to UN
pydicom.config.convert_wrong_length_to_UN = True

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

    # Standard orientation
    #sitk_image = sitk.DICOMOrient(sitk_image, 'LPS')
    numpy_image = sitk.GetArrayFromImage(sitk_image).astype(float)

    # --- FIX: Initialize the return variable with the original image ---
    # This ensures sitk_image_suv always has a value.
    # It will be overwritten later only if the modality is PET and conversion succeeds.
    sitk_image_suv = sitk_image

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
                # Skip problematic tags and optionally log
                print(f"Warning: skipping tag {tag_str} due to parsing error: {e}")

    # --- Determine modality ---
    modality = full_metadata.get('0008|0060', None)

    # --- PET conversion to SUVbw ---
    if modality == "PT":
        try:
            def get_tag(tag_str, default=None):
                return full_metadata.get(tag_str, default)

            weight_kg = float(get_tag('0010|1030', 0))
            injected_dose_bq = float(get_tag('0018|1074', 0))
            half_life_s = float(get_tag('0018|1075', 0))
            scan_time_str = str(get_tag('0008|0031', '000000')) # Series Time
            injection_time_str = str(get_tag('0018|1072', '000000')) # Radiopharmaceutical Start Time

            if injected_dose_bq == 0 or weight_kg == 0 or half_life_s == 0:
                raise ValueError("Missing essential PET metadata for SUV conversion (Dose, Weight, or HalfLife).")

            # Convert HHMMSS to seconds
            def hhmmss_to_seconds(hhmmss):
                hhmmss = str(hhmmss).split('.')[0].zfill(6)
                h = int(hhmmss[0:2]); m = int(hhmmss[2:4]); s = int(hhmmss[4:6])
                return h*3600 + m*60 + s

            elapsed_seconds = hhmmss_to_seconds(scan_time_str) - hhmmss_to_seconds(injection_time_str)
            
            if elapsed_seconds < 0:
                print("Warning: Scan time appears to be before injection time. Check DICOM tags. Assuming no decay.")
                elapsed_seconds = 0
                
            dose_at_acq = injected_dose_bq * (2 ** (-elapsed_seconds / half_life_s))
            
            if dose_at_acq == 0:
                raise ValueError("Dose at acquisition time is zero, cannot compute SUV.")

            # SUVbw conversion
            suv_factor = (weight_kg * 1000) / dose_at_acq
            numpy_image = numpy_image * suv_factor
            
            # Create the new SimpleITK image with SUV values
            sitk_image_suv = sitk.GetImageFromArray(numpy_image)
            sitk_image_suv.CopyInformation(sitk_image) # Preserve original geometry

        except Exception as e:
            print(f"Warning: Could not convert PET to SUVbw: {e}. Returning image with original values.")
            # If conversion fails, numpy_image will still hold the original values because
            # the SUV calculation failed, and sitk_image_suv was already set to the original sitk_image.

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
        print(f"Warning: Unknown interpolator '{method_name}'. Defaulting to linear.")
        return sitk.sitkLinear

def align_pet_to_ct(pet_image: sitk.Image, ct_image: sitk.Image, interpolation_method: str = 'bspline'):
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
    print(f"CT Size: {ct_image.GetSize()}, Origin: {ct_image.GetOrigin()}, Direction: {ct_image.GetDirection()}")
    print(f"PET Size: {pet_image.GetSize()}, Origin: {pet_image.GetOrigin()}, Direction: {pet_image.GetDirection()}")
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
    print(f"Aligned PET Size: {resampled_pet_image.GetSize()}, Origin: {resampled_pet_image.GetOrigin()}, Direction: {resampled_pet_image.GetDirection()}")

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