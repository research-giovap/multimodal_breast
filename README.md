# LesionLoop Breast

**Lesion-aware multimodal interactive segmentation of breast cancer on [18F]FDG PET/CT**

LesionLoop Breast is a desktop application for the interactive segmentation of breast lesions on [18F]FDG PET/CT. The operator places a single click or scribble on the lesion. A 2.5D multimodal network then segments the lesion on that slice, and a self-calibrated slice-by-slice propagation extends the result to 3D. The application reports the metabolic tumour volume (MTV), SUVmax, SUVmean and total lesion glycolysis (TLG), and exports the mask in NIfTI format.

The application runs the procedure validated in the accompanying article. It produces, voxel by voxel, the masks evaluated in the study.

> Pasini G, Chauvie S, Cuttone S, Buschiazzo A, Costa M, Russo G, Stefano A. *Lesion-aware multimodal interactive segmentation of breast cancer on [18F]FDG PET/CT: a domain-specific deep learning ensemble improves clinical specificity over 3D promptable foundation models.* (submitted)

**Intended use.** This software is intended for research only. It is not a medical device and must not be used for diagnosis or treatment decisions.

---

## Method in brief

- **Input.** The network receives a 2.5D patch of 7 × 224 × 224 pixels on a 1.171875 × 1.171875 × 3.0 mm grid. The patch contains three CT slices (window 40/400 HU), three PET slices (min–max normalised SUV) and a prompt map, which is an exponential distance map of the operator's click or scribble.
- **Model.** The default model is a soft-voting ensemble of a five-level attention U-Net and a multi-branch network (residual U-Net + attention U-Net). The probabilities of the two networks are averaged before thresholding at 0.5.
- **3D propagation.** The segmentation is propagated slice by slice in both directions. The stopping criteria are computed for each examination on the prompt slice, without thresholds tuned on data:
  - network confidence (Otsu threshold);
  - PET contrast at half maximum above the soft-tissue background defined on CT;
  - minimal 2D support;
  - slice-to-slice geometric continuity.
- **Re-anchoring.** The propagation is then repeated from the lesion SUVmax, and the largest 3D connected component is kept.

The network is lesion-aware. A click a few millimetres outside the lesion still recovers the lesion. A click on non-tumoural uptake (for example myocardium or kidney) or on normal tissue usually returns an empty mask.

## Requirements

- Windows 10/11 or Linux, 64-bit.
- Python **3.12** (tested with 3.12.4).
- An NVIDIA GPU with CUDA is optional. A 3D segmentation takes about 1.7 s per lesion on an RTX 3070 and about 3 s on a recent CPU.
- About 1 GB of disk space for the Python environment, plus the PyTorch installation.

## Installation

```bash
git clone <repository URL>
cd <repository folder>

# 1. Create and activate a virtual environment with Python 3.12
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# 2. Install PyTorch (tested with 2.11.0)
#    NVIDIA GPU (CUDA 12.8):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
#    CPU only:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the other dependencies
pip install -r requirements.txt
```

The trained weights are included in `segmentation/models/` (about 165 MB).

## Usage

```bash
python main.py
```

1. **Load CT Folder.** Select the folder that contains the CT DICOM series.
2. **Load PET Folder.** Select the folder that contains the PET DICOM series. The PET is converted to body-weight SUV from the DICOM header (injected dose, patient weight, injection and series times), aligned to the CT and prepared for inference.
3. Scroll to a slice that contains the lesion and select **Draw**. Place a click or a short scribble on the lesion. The brush size can be adjusted.
4. **Segment Volumetric (3D)** segments the whole lesion. **Segment Slice (2D)** segments only the current slice.
5. The **Quantitative Summary** panel shows MTV (mL), number of slices, SUVmax, SUVmean and TLG (g).
6. **Save 3D Mask (NIfTI)** writes the mask on the CT geometry, so it can be opened together with the original CT.

The ensemble is selected by default. The four single networks remain selectable for comparison.

**Mouse controls**

| Action | Control |
| --- | --- |
| Change slice | mouse wheel |
| CT window/level | left drag (Nav mode) |
| PET window | Ctrl + left drag (Nav mode) |
| Zoom | right drag |
| Pan | middle drag |
| Draw / erase prompt | left drag (Draw / Erase mode) |
| Hide / show side panel | ☰ button |

**Notes on the input data**

- CT and PET must be separate DICOM series of the same examination.
- SUV conversion requires the Radiopharmaceutical Total Dose, the Patient Weight, the Radionuclide Half Life, the Radiopharmaceutical Start Time and the Series Time. If any of them is missing, the PET is displayed with its original values and the SUV statistics are not meaningful.
- The method was developed and validated for primary breast lesions. It was not designed for nodal or distant lesions, and it handles one lesion per prompt.

## Use without the graphical interface

The segmentation can be called from other Python code. The modules in `segmentation/` and `preprocessing/` do not depend on the interface or on PyQt5.

```python
import numpy as np
import torch
from preprocessing import preprocessing as prep
from segmentation.inference import load_method, segment_volume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_method("segmentation/models", "ensemble", device)        # validated ensemble

# Load the DICOM series (PET in body-weight SUV) and align the PET to the CT grid
sitk_ct, ct, _, _ = prep.get_image("path/to/CT_series")
sitk_pet, _, _, _ = prep.get_image("path/to/PET_series")
pet_on_ct, pet_array = prep.align_pet_to_ct(sitk_pet, sitk_ct)
ai_ct, ai_pet, scale_info = prep.prepare_volumes_for_inference(sitk_ct, ct, pet_on_ct, pet_array)

# Prompt: binary 2D mask (click or scribble) on slice z of the CT grid
z = 120
prompt = np.zeros(ct.shape[1:], np.uint8)
prompt[250:254, 300:304] = 1

mask_ai, _ = segment_volume(model, device, ai_ct, ai_pet, scale_info, prompt, z)
mask_ct = prep.resample_full_mask_back_to_viewer(mask_ai.astype(np.uint8),
                                                 scale_info["sitk_resampled_ref"], sitk_ct)  # CT geometry
```

`get_image`, `align_pet_to_ct` and `prepare_volumes_for_inference` need to run only once per examination. `segment_volume` can then be called for every prompt. Load the model once and reuse it.

## Repository structure

```
main.py                        entry point
gui/
  viewerPETCT.py               main window
  customQlabel.py              image widget (zoom, pan, coordinate mapping)
preprocessing/
  preprocessing.py             DICOM loading, SUV, PET-to-CT alignment, resampling, patch extraction
segmentation/
  inference.py                 validated 2D and 3D segmentation procedure
  model_defs.py                network architectures and weight registry
  models/
    unet_2.pth                 residual U-Net (5 levels)
    unet_3.pth                 attention U-Net (4 levels)
    unet_4.pth                 attention U-Net (5 levels)       ┐ validated
    unet_5.pth                 multi-branch network             ┘ ensemble
```

The constants of the propagation in `segmentation/inference.py` are those evaluated in the article. If they are changed, the method must be validated again.

## Citation

If you use LesionLoop Breast, please cite the accompanying article (reference to be updated on publication).

## Licence

The code and the trained model weights are released under the [Apache License 2.0](LICENSE). See also [NOTICE](NOTICE).

The segmentation and pre-processing modules (`segmentation/`, `preprocessing/`) and the weights depend only on packages with permissive licences: NumPy, SciPy, scikit-image, PyTorch (BSD), SimpleITK (Apache 2.0) and pydicom (MIT).

Only the graphical interface (`gui/`, `main.py`) uses PyQt5, which Riverbank Computing distributes under the GPL v3 or a commercial licence. Redistributing the application together with PyQt5 must comply with the PyQt5 licence. This does not apply to software that uses only the segmentation and pre-processing modules.

## Acknowledgements

This work was funded by the PR FESR Piemonte 2021/2027 programme within the project BIS – The Breast Integrated Solution (SWIch call). It was carried out by the Institute of Bioimaging and Complex Biological Systems of the National Research Council of Italy (IBSBC-CNR), the Azienda Ospedaliera S. Croce e Carle (Cuneo) and Tecnologie Avanzate T.A. S.r.l.
