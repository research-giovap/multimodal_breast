import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import preprocessing.preprocessing as preprocessing
import pandas as pd
import matplotlib.pyplot as plt

z_dim = []
y_dim = []
x_dim = []
z_min = []
y_min = []
x_min = []
z_max = []
y_max = []
x_max = []
directory = 'dataset_training'
patients = [p for p in os.listdir(directory)]
patient = [[p for p in patients if p =='29'][0]]
p_list = []
for elem in patients:
    print(f"Processing patient {elem}...")
    series_name = [s_name for s_name in os.listdir(os.path.join(directory, elem, elem)) if 'MASK' in s_name][0]
    path = os.path.join(directory, elem, elem, series_name)
    maskname = [m_name for m_name in os.listdir(path) if m_name.endswith('.nii') or m_name.endswith('.nii.gz')][0]
    path = os.path.join(path, maskname)
    print(f"  Loading from {path}...")
    sitk_mask, numpy_mask = preprocessing.get_mask(path)
    resampled_sitk_mask, resampled_np_mask = preprocessing.resample(sitk_mask, new_spacing=(1.171875, 1.171875, 3.0), interpolation_method='nearest')
    # bbox_dims = preprocessing.get_bbox_dimension(resampled_np_mask)
    bbox_slices, bbox_dimensions, box_coordinates = preprocessing.bbox_full_extent(resampled_np_mask, resampled_np_mask, display=False)
    
    z_dim.append(bbox_dimensions[0])
    y_dim.append(bbox_dimensions[1])
    x_dim.append(bbox_dimensions[2])
    z_min.append(box_coordinates[0])
    y_min.append(box_coordinates[1])
    x_min.append(box_coordinates[2])
    z_max.append(box_coordinates[3])
    y_max.append(box_coordinates[4])
    x_max.append(box_coordinates[5])
    p_list.append(elem)

df = pd.DataFrame({'Patient': p_list, 
                   'Z dim': z_dim, 
                   'Y dim': y_dim, 
                   'X dim': x_dim, 
                   'Z min': z_min, 
                   'Y min': y_min,
                   'X min': x_min,
                   'Z max': z_max,
                   'Y max': y_max,
                   'X max': x_max})
df.to_excel('mask_sizing_2.xlsx', index=False)