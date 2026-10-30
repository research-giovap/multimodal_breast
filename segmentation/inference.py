"""
inference.py - The validated segmentation method, shared by the GUI.

This module contains exactly the procedure evaluated with validate_models.py (model name suffix
"_adap3") on the internal validation set and on the external QIN-Breast test:

  * model: soft-voting ensemble of unet_4.pth (AttentionUNet25D_depth5) and unet_5.pth
    (MultiBranchUNet); probabilities are averaged before the 0.5 threshold;
  * 2D: one inference on the 224x224 patch centred on the operator's prompt;
  * 3D: bidirectional slice-by-slice propagation with self-calibrated
    stopping, computed on the prompt slice of each examination:
      - network confidence: Otsu threshold of the probability map (bounded by mu - 3 sigma, min 0.5);
      - minimal 2D support of 4 pixels;
      - PET contrast: half-maximum level of the lesion contrast above the soft-tissue background
        (median and MAD of the CT-windowed soft tissue of the patch);
      - geometric continuity: IoU >= e^-2 (unless the slice is a concentric polar taper) and no
        expansion > e^(1/e) after the area fell below e^(-1/e) of the largest area;
    a second propagation restarts from the SUVmax of the component connected to the prompt slice;
    the largest 3D connected component is kept.

Do not change the constants unless the method is re-validated.
"""

import os

import numpy as np
import torch
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import resize

try:
    from preprocessing import preprocessing as prep
except ImportError:          # flat layout
    import preprocessing as prep
try:
    from segmentation.model_defs import MODEL_REGISTRY, DEFAULT_MODEL_CONFIG
except ImportError:
    from model_defs import MODEL_REGISTRY, DEFAULT_MODEL_CONFIG

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
THR = 0.5                    # probability threshold
MIN_SUPPORT = 4              # minimal 2D support (2 x 2 pixels)
RESEED_RADIUS = 3            # radius (pixels) of the seed placed at the SUVmax

VALIDATED_ENSEMBLE = ("unet_4.pth", "unet_5.pth")
VALIDATED_ENSEMBLE_LABEL = "Ensemble unet_4 + unet_5 (validated)"


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class EnsembleModel(torch.nn.Module):
    """Soft voting: averages the members' probabilities and returns the logit of the mean,
    so that sigmoid + 0.5 threshold works unchanged. In the 3D propagation the average is
    taken at every slice, so each slice's prompt comes from the ensemble."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        p = torch.stack([torch.sigmoid(m(x)) for m in self.members], 0).mean(0)
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)


def load_single_model(path, device):
    name = os.path.basename(path)
    cfg = MODEL_REGISTRY.get(name, DEFAULT_MODEL_CONFIG)
    model = cfg["class"](**cfg["params"])
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_method(model_folder, spec, device):
    """spec: a .pth file name, or 'ensemble' for the validated ensemble."""
    if spec == "ensemble":
        members = [load_single_model(os.path.join(model_folder, f), device) for f in VALIDATED_ENSEMBLE]
        return EnsembleModel(members).to(device).eval()
    return load_single_model(os.path.join(model_folder, spec), device)


def ensemble_available(model_folder):
    return all(os.path.exists(os.path.join(model_folder, f)) for f in VALIDATED_ENSEMBLE)


@torch.no_grad()
def predict(model, x, device):
    """x: (7,H,W) or (B,7,H,W) numpy -> probabilities (H,W) or (B,H,W)."""
    single = x.ndim == 3
    t = torch.from_numpy(np.ascontiguousarray(x[None] if single else x)).float().to(device)
    p = torch.sigmoid(model(t))[:, 0].cpu().numpy()
    return p[0] if single else p


# --------------------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------------------
def segment_slice(model, device, ai_ct, ai_pet, scale_info, seed_viewer_2d, z_viewer):
    """Single-slice segmentation. Returns (prediction patch, metadata) for
    preprocessing.reconstruct_and_resample_back_optimized."""
    x, meta = prep.run_inference_on_preprocessed(ai_ct, ai_pet, scale_info, seed_viewer_2d, z_viewer)
    pred = (predict(model, x, device) > THR).astype(np.uint8)
    return pred, meta


# --------------------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------------------
def _slice_iou(a, b):
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return (inter / union) if union > 0 else 0.0


def _bbox3(a):
    nz = np.nonzero(a)
    if not len(nz[0]):
        return None
    return tuple(slice(int(x.min()), int(x.max()) + 1) for x in nz)


def _inclusion_ratio(a, b):
    """Overlap (Szymkiewicz-Simpson) coefficient |a & b| / min(|a|, |b|)."""
    inter = int(np.count_nonzero(a & b))
    denom = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    return (inter / denom) if denom > 0 else 0.0


def _cc_touching(vol, z0, ref2d):
    """Keeps the 3D connected components of vol that overlap ref2d on slice z0."""
    if not vol.any() or not ref2d.any():
        return np.zeros_like(vol)
    bb = _bbox3(vol)
    lab, n = ndimage.label(vol[bb] > 0)
    if n == 0 or not (bb[0].start <= z0 < bb[0].stop):
        return np.zeros_like(vol)
    seed_lab = lab[z0 - bb[0].start][ref2d[bb[1], bb[2]] > 0]
    keep = np.unique(seed_lab[seed_lab > 0])
    out = np.zeros_like(vol)
    if len(keep):
        out[bb] = np.isin(lab, keep).astype(vol.dtype)
    return out


def propagate_adaptive(model, device, ai_ct, ai_pet, start_z, seed2d, progress=None):
    """Bidirectional propagation with self-calibrated stopping (validate_models.py: _adap3).
    All bounds are computed on the prompt slice of the examination; see the module docstring."""
    d, h, w = ai_ct.shape
    full = np.zeros((d, h, w), np.uint8)
    x0, meta0 = prep.run_inference_direct_ai_space(ai_ct, ai_pet, seed2d, start_z)
    prob0 = predict(model, x0, device)
    pred0 = (prob0 > THR).astype(np.uint8)
    if pred0.sum() < MIN_SUPPORT:
        return full
    start = prep.paste_patch_into_slice(pred0, meta0)
    if start.sum() < MIN_SUPPORT:
        return full
    full[start_z] = start

    # 1. network confidence: Otsu threshold, bounded by the lower tail of the predicted pixels
    nz_probs = prob0[prob0 > 1e-4]
    if nz_probs.size >= 10:
        try:
            p_otsu = float(threshold_otsu(nz_probs))
        except Exception:
            p_otsu = THR
    else:
        p_otsu = THR
    pos_probs = prob0[pred0 > 0]
    conf_bound = min(p_otsu, max(float(pos_probs.mean()) - 3.0 * float(pos_probs.std()), THR))

    # 2. soft-tissue background of the patch (CT window > -156 HU, outside the prediction)
    ct_c, pet_c = x0[1], x0[4]
    bg_vals = pet_c[(pred0 == 0) & (ct_c > 0.01) & (pet_c > 0)]
    if bg_vals.size < 10:
        bg_vals = pet_c[(pred0 == 0) & (pet_c > 0)]
    if bg_vals.size > 0:
        bg_med = float(np.median(bg_vals))
        bg_mad = float(1.4826 * np.median(np.abs(bg_vals - bg_med)))
    else:
        bg_med, bg_mad = 0.0, 0.0

    les = ai_pet[start_z][start > 0]
    start_area = int(start.sum())
    inv_e2 = float(np.exp(-2.0))            # ~0.135
    inv_e_1_e = float(np.exp(-1.0 / np.e))  # ~0.692
    exp_1_e = float(np.exp(1.0 / np.e))     # ~1.445
    states = {s: {"active": True, "cur": start, "z": start_z, "prev_area": start_area, "max_area": start_area,
                  "peak_pet": float(les.max()), "ref_med_pet": float(np.median(les))} for s in (1, -1)}

    while states[1]["active"] or states[-1]["active"]:
        dirs, xs, metas, nzs = [], [], [], []
        for direction in (1, -1):
            st = states[direction]
            if not st["active"]:
                continue
            nz = st["z"] + direction
            if not (0 <= nz < d):
                st["active"] = False
                continue
            x, meta = prep.run_inference_direct_ai_space(ai_ct, ai_pet, st["cur"], nz)
            dirs.append(direction); xs.append(x); metas.append(meta); nzs.append(nz)
        if not xs:
            break
        if progress is not None:
            progress(f"slices {', '.join(str(n) for n in nzs)}")
        probs = predict(model, np.stack(xs), device)
        for idx, direction in enumerate(dirs):
            st = states[direction]
            nz = nzs[idx]
            prob = probs[idx]
            p = (prob > THR).astype(np.uint8)
            if p.sum() < MIN_SUPPORT or float(prob[p > 0].mean()) < conf_bound:
                st["active"] = False
                continue
            clean = prep.filter_disconnected_blobs(st["cur"], prep.paste_patch_into_slice(p, metas[idx]))
            cur_area = int(clean.sum())
            if cur_area < MIN_SUPPORT:
                st["active"] = False
                continue
            # geometric continuity (a concentric polar taper is not a jump)
            tapering = (cur_area <= st["prev_area"]) and (_inclusion_ratio(st["cur"], clean) >= inv_e_1_e)
            if _slice_iou(st["cur"], clean) < inv_e2 and not tapering:
                st["active"] = False
                continue
            # PET contrast: half-maximum of the lesion contrast above the soft-tissue background
            vals = ai_pet[nz][clean > 0]
            cur_max = float(vals.max())
            st["peak_pet"] = max(st["peak_pet"], cur_max)
            st["ref_med_pet"] = max(st["ref_med_pet"], float(np.median(vals)))
            ex_med = max(st["ref_med_pet"] - bg_med, 0.0)
            ex_peak = max(st["peak_pet"] - bg_med, 0.0)
            contrast = np.sqrt(ex_med * np.sqrt(ex_med * ex_peak))
            thr = max(bg_med + 0.5 * contrast, bg_med + 2.0 * bg_mad)
            if st["peak_pet"] > (bg_med + 2.0 * bg_mad) and cur_max < thr:
                st["active"] = False
                continue
            # sudden expansion after necking
            if st["prev_area"] < inv_e_1_e * st["max_area"] and cur_area > exp_1_e * st["prev_area"]:
                st["active"] = False
                continue
            full[nz] = clean
            st["prev_area"] = cur_area
            st["max_area"] = max(st["max_area"], cur_area)
            st["cur"] = clean
            st["z"] = nz
    return full


def segment_volume(model, device, ai_ct, ai_pet, scale_info, seed_viewer_2d, z_viewer, progress=None):
    """Volumetric segmentation in AI space. Returns (AI mask, info dict).
    Stages: 1) propagation from the operator's prompt; 2) SUVmax of the result, within the component
    connected to the prompt slice; 3) propagation from a seed at the
    SUVmax; 4) largest 3D connected component."""
    d, h, w = ai_ct.shape
    z_ai = int(np.clip(round(z_viewer * scale_info["scale_z"]), 0, d - 1))
    seed_ai = resize(seed_viewer_2d, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    info = {"stage1_slices": 0, "suvmax_reseed": False}
    run = propagate_adaptive

    say = (lambda stage: (lambda t: progress(f"{stage}: {t}"))) if progress else (lambda stage: None)
    vol1 = run(model, device, ai_ct, ai_pet, z_ai, seed_ai, progress=say("Stage 1 - propagation from the prompt"))
    info["stage1_slices"] = int(vol1.any(axis=(1, 2)).sum())
    if not vol1.any():
        return vol1, info
    final = vol1
    search = vol1
    if vol1[z_ai].any():
        cc = _cc_touching(vol1, z_ai, vol1[z_ai])
        search = cc if cc.any() else vol1
    suv = prep.find_suv_max_centroid(ai_pet, search)
    if suv is not None:
        new_seed = prep.create_local_seed_mask(ai_ct.shape, suv, radius=RESEED_RADIUS)
        final = run(model, device, ai_ct, ai_pet, suv[0], new_seed,
                    progress=say("Stage 2 - propagation from the SUVmax"))
        info["suvmax_reseed"] = True
    if progress:
        progress("Stage 3 - 3D connectivity")
    bb = _bbox3(final)
    if bb is not None:
        sub = prep.keep_largest_connected_component_3d(final[bb])
        final = np.zeros_like(final)
        final[bb] = sub
    return final.astype(np.uint8), info
