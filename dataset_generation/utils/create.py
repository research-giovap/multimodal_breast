import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import preprocessing.preprocessing as preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time



directory = 'dataset_validation'
patients = [p for p in os.listdir(directory)]
# patient = [[p for p in patients if p =='29'][0]]
p_list = []
dataset_elem = []
label = []
weights_label = []
for elem in patients:
    print(f"Processing patient {elem}...")
    series_name_mask = [s_name for s_name in os.listdir(os.path.join(directory, elem, elem)) if 'MASK' in s_name][0]
    path_mask = os.path.join(directory, elem, elem, series_name_mask)
    maskname = [m_name for m_name in os.listdir(path_mask) if m_name.endswith('.nii') or m_name.endswith('.nii.gz')][0]
    path_mask = os.path.join(path_mask, maskname)

    series_name_CT = [ct_name for ct_name in os.listdir(os.path.join(directory, elem, elem)) if 'CT' in ct_name and ('PT' not in ct_name or 'PET' not in ct_name)][0]
    path_CT = os.path.join(directory, elem, elem, series_name_CT)

    series_name_PT = [pt_name for pt_name in os.listdir(os.path.join(directory, elem, elem)) if ('PT' in pt_name or 'PET' in pt_name)][0]
    path_PT = os.path.join(directory, elem, elem, series_name_PT)
    print(f"  Loading CT from {path_CT}...")
    print(f"  Loading CT from {path_PT}...")
    print(f"  Loading Mask from {path_mask}...")
    sitk_ct, numpy_ct, modality_ct, metadata_ct = preprocessing.get_image(path_CT)
    resampled_np_ct = preprocessing.resample(sitk_ct, numpy_ct, new_spacing=(1.171875, 1.171875, 3.0), interpolation_method='linear')
    
    sitk_pet, numpy_pet, modality_pt, metadata_pt = preprocessing.get_image(path_PT)
    aligned_pet_image, aligned_pet_array = preprocessing.align_pet_to_ct(sitk_pet, sitk_ct, interpolation_method='linear')
    resampled_np_pet = preprocessing.resample(aligned_pet_image, aligned_pet_array, new_spacing=(1.171875, 1.171875, 3.0), interpolation_method='linear')
    
    sitk_mask, numpy_mask = preprocessing.get_mask(path_mask)
    aligned_mask_image, aligned_mask_array = preprocessing.align_pet_to_ct(sitk_mask, sitk_ct, interpolation_method='nearest')
    resampled_np_mask = preprocessing.resample(aligned_mask_image, aligned_mask_array, new_spacing=(1.171875, 1.171875, 3.0), interpolation_method='nearest')
    # preprocessed_ct, preprocessed_ct_mask = preprocessing.preprocess_ct(resampled_np_ct)
    preprocessed_ct = resampled_np_ct
    vol_ct_norm = preprocessing.apply_window(preprocessed_ct, center=40, width=400)
    vol_pet_norm = preprocessing.min_max_normalize(resampled_np_pet)

    
    patches_ct_propagation, patches_pet_propagation, scribble_mask_propagation, patches_mask_propagation = preprocessing.sample_patches_2d_propagation(vol_ct_norm, vol_pet_norm, resampled_np_mask, patch_size=(224,224))
    weights_propagation = preprocessing.compute_mask_percentage(patches_mask_propagation)
    patches_ct_init, patches_pet_init, patches_mask_init, scribble_mask_init= preprocessing.sample_patches_25d_initialization(vol_ct_norm, 
                                                                                                                              vol_pet_norm, 
                                                                                                                              resampled_np_mask, 
                                                                                                                              patch_size=(224, 224),
                                                                                                                              scribble_min_length=20, 
                                                                                                                              scribble_max_length=300,
                                                                                                                              scribble_dilation=1, 
                                                                                                                              prompt_sigma=20, 
                                                                                                                              min_coverage=0.3,
                                                                                                                              transformation = 'SDF')
    weights_init = np.zeros((patches_ct_init.shape[0]), dtype = np.int8)

    vec_init = np.zeros((patches_ct_init.shape[0],), dtype=np.int8)
    vec_propagation = np.ones((patches_ct_propagation.shape[0],), dtype=np.int8)
    
    patches_ct = np.concatenate((patches_ct_init, patches_ct_propagation), axis=0)
    vec_all = np.concatenate((vec_init, vec_propagation), axis=0)
    weights_all = np.concatenate((weights_init, weights_propagation), axis=0)
    patches_pet = np.concatenate((patches_pet_init, patches_pet_propagation), axis=0)

    patches_scribble_mask = np.concatenate((scribble_mask_init, scribble_mask_propagation), axis=0)
    patches_mask = np.concatenate((patches_mask_init, patches_mask_propagation), axis=0)
 

    multimodal_volume = np.concatenate((patches_ct, patches_pet), axis=1)  # concatenate along channel dimension
    patches_scribble_mask = np.expand_dims(patches_scribble_mask, axis=1)  # add channel dimension
    multimodal_volume = np.concatenate((multimodal_volume, patches_scribble_mask), axis=1)  # concatenate along channel dimension
    

    os.mkdir(f'N:\multimodal_volumes_pet_ct\{elem}')
    # os.mkdir(f'N:\multimodal_volumes_pet_ct\{elem}'+'_images')
    output_dir = os.path.join('N:\multimodal_volumes_pet_ct', elem)
    # output_dir_images = os.path.join('N:\multimodal_volumes_pet_ct', elem+'_images')
    preprocessing.save_multimodal_volumes_slices(multimodal_volume, patches_mask, output_dir)
    # preprocessing.visualize_multimodal_triplets(
    # multimodal_volume, patches_mask, vec_all, weights_all,
    # save_dir=output_dir_images,
    # cmap_ct='gray', cmap_pet='hot', cmap_mask='gray',
    # prefix="triplet")
    for i in range(multimodal_volume.shape[0]):
        dataset_elem.append(f"{elem}_slice_{i:03d}")
        label.append(vec_all[i])
        weights_label.append(weights_all[i])

df = pd.DataFrame({'id': dataset_elem, 'label_init_vs_prop': label, 'presence': weights_label})
df.to_excel('N:\multimodal_volumes_pet_ct\labels.xlsx', index=False)

 


   
 
   


    