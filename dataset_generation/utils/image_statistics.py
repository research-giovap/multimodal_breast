
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import preprocessing.preprocessing as preprocessing
import pandas as pd

z_pix = []
y_pix = []
x_pix = []
directory = 'dataset_training'
patients = [p for p in os.listdir(directory)]
p_list = []
for elem in patients:
    print(f"Processing patient {elem}...")
    series_name = [s_name for s_name in os.listdir(os.path.join(directory, elem, elem)) if 'CT' in s_name and 'PT' not in s_name][0]
    path = os.path.join(directory, elem, elem, series_name)
    print(f"  Loading from {path}...")
    sitk_image, numpy_image, modality, metadata = preprocessing.get_image(path)
    z_pix.append(sitk_image.GetSpacing()[2])
    y_pix.append(sitk_image.GetSpacing()[1])
    x_pix.append(sitk_image.GetSpacing()[0])
    p_list.append(elem)

df = pd.DataFrame({'Patient': p_list, 'Z spacing (mm)': z_pix, 'Y spacing (mm)': y_pix, 'X spacing (mm)': x_pix})
df.to_excel('image_spacing.xlsx', index=False)
