import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation
from collections import deque
import shutil
import subprocess
from pathlib import Path
import nibabel as nib
from itertools import permutations

'''
Description:
This code will generate projection binary image from CT image dataset.

Usage:
1. This code require envrionment that has the following toolkit:
    use "conda activate totalseg" first. I've download all required toolkit in
    the conda environment "totalseg"
2. Run the code:
    python3 /data/weiyuq/Code/binary_proj_img_generator_dataset.py
'''


# =========================
# Setup the path and workplace
# =========================
main_dir = Path("[Path to ct folder]")

output_dir_hi = Path("[High energy projection output directory]")
output_dir_low = Path("[Low energy projection output directory]")
output_dir_path = Path("[Path length projection output directory]")

work_dir = Path("[A work directory for totalsegmentator]") # Bone seg output path




# "fast" will cause space to be 3mm. Low resolution
if_fast = False

vmin = 0
vmax = 620

out_h = 512
out_w = 512

air_thr = 100
bone_tis_thr = 1300
ir_thr = air_thr

bone_density = 1.920 # g/cm^3
tissue_density = 1.060 # g/cm^3

# Set up the attenuation coefficient
bone_mask_attenuation_l = 0.2229    # cm^2/g
tissue_mask_attenuation_l = 0.1823  # cm^2/g

bone_mask_attenuation_h = 0.148     # cm^2/g
tissue_mask_attenuation_h = 0.1492  # cm^2/g

tissue_value_low = tissue_density * tissue_mask_attenuation_l
bone_value_low = bone_density * bone_mask_attenuation_l

tissue_value_high = tissue_density * tissue_mask_attenuation_h
bone_value_high = bone_density * bone_mask_attenuation_h


# =========================
# Extract img data and apply the threshold
# =========================
def extract_data_thr(dcm_path: Path):
    '''
    Extract data from dicom for single slice and apply the threshold to prepare
    for couch removal.

    Input:
    dcm_path: DICOM path of single slice

    Output:
    img: image data
    '''
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.int16)

    '''
    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0, vmax=None)
    plt.title(f"Original image")
    plt.axis("off")
    plt.show()
    '''
    
    return img


# =========================
# Remove Couch
# =========================
def remove_couch(img):
    '''
    Input:
    img: the np image data    

    Output:
    img_good: image data after couch is removed
    no_hole_mask: body mask (filled holes)
    '''
    img_thr = img.copy()
    img_thr[img_thr < vmax] = 0

    mask_base = (img_thr > 0)      # True = nonzero region
    mask = binary_erosion(mask_base, iterations=1)  # Erosion of the mask's edge.
    H, W = mask.shape

    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            if labels[r, c] != 0:
                continue

            # found a new component
            current_label += 1
            q = deque([(r, c)])
            labels[r, c] = current_label

            while q:
                rr, cc = q.popleft()
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            q.append((nr, nc))

    patient_mask = (labels == 1)
    img_patient = img_thr.copy()
    img_patient[labels != 1] = 0

    no_hole_mask = img_patient.copy()
    no_hole_mask = binary_fill_holes(no_hole_mask)
    img_good = img.copy()
    img_good = img_good * no_hole_mask

    return img_good


# =========================
# Run TotalSegmentator (total task)
# =========================
def run_totalseg_total(dicom_dir: Path, out_dir: Path, fast: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = shutil.which("TotalSegmentator")
    if exe is None:
        raise RuntimeError("Cannot find TotalSegmentator in PATH. Did you conda activate totalseg?")

    #cmd = [exe, "-i", str(dicom_dir), "-o", str(out_dir)]
    cmd = [exe, "-i", str(dicom_dir), "-o", str(out_dir), "--device", "gpu"]
    if fast:
        cmd += ["--fast"]

    subprocess.run(cmd, check = True)



# =========================
# Get Bone Mask
# =========================
def get_bone_mask(dicom_dir: Path, ts_total_dir: Path, if_fast: bool, bone_out: Path):
    '''
    This only need to run one time for each CT.
    '''
    # 1) Run TotalSegmentator（total task）
    run_totalseg_total(dicom_dir, ts_total_dir, fast=if_fast)

    # 2) get bone_mask.nii.gz
    build_bone_mask(
        ts_total_dir = ts_total_dir,
        out_path = bone_out,
        bone_keys = None,
        fill_holes_2d = False,
        save_preview_png = False
    )

    print("\n Done.")
    print(f"Bone mask saved at: {bone_out}")



# =========================
# Sort DICOM slices (robust)
# =========================
def sort_dicom_slices(dicom_dir: Path):
    return sorted(Path(dicom_dir).glob("*.dcm"))





# =========================
# Load bone_mask.nii.gz and align to (Z,H,W)
# =========================
def load_bone_mask_as_zyx(bone_nii_path: Path, target_shape_zyx):
    """
    target_shape_zyx = (Z,H,W)
    return mask_zyx bool array with shape (Z,H,W)
    """
    img = nib.load(str(bone_nii_path))
    data = img.get_fdata()
    mask = data > 0

    for perm in permutations([0, 1, 2]):
        m = np.transpose(mask, perm)
        if m.shape == target_shape_zyx:
            #print(f"[Mask align] NIfTI axes permute {perm} -> (Z,H,W)")
            return m.astype(bool)

    raise ValueError(f"Cannot align NIfTI shape {mask.shape} to target {target_shape_zyx}")



# =========================
# Clean everything after demo
# =========================
def safe_cleanup(dir_path: Path):
    """
    Delete everything under work_dir after demo.
    """
    dir_path = Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors = True)
    print(f"[Clean] deleted: {dir_path}")


# =========================
# Helpers for building bone mask
# =========================
def load_mask_nii(path: Path):
    """
    Read a single NIfTI mask file from TotalSeg output.

    Output:
      mask_bool: np.ndarray (bool)
      ref_img:   nib.Nifti1Image (keeps affine/header for saving)
    """
    img = nib.load(str(path))
    # memory-friendly: keep original dtype, avoid float64 from get_fdata()
    data = np.asanyarray(img.dataobj)
    mask = data > 0
    return mask, img


# =========================
# Helper function that we save bone map to output directory
# =========================
def save_mask_like(ref_img: nib.Nifti1Image, mask: np.ndarray, out_path: Path):
    """
    Save mask using the reference NIfTI's affine/header.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_img = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))
    print(f"[Saved] {out_path}")


# =========================
# Build the bone map
# =========================
def build_bone_mask(
    ts_total_dir: Path,
    out_path: Path,
    bone_keys=None,
    fill_holes_2d: bool = False,
    save_preview_png: bool = True,
):
    """
    Combine multiple bone-related TotalSeg masks into one union bone mask.

    Inputs:
      ts_total_dir: folder containing TotalSeg total-task outputs (*.nii.gz)
      out_path:     output bone_mask.nii.gz path
      bone_keys:    list of keywords to match bone parts in filenames
      fill_holes_2d: fill holes slice-by-slice (improves mask quality)
      save_preview_png: save mid-slice preview image
    """
    ts_total_dir = Path(ts_total_dir)
    out_path = Path(out_path)

    if bone_keys is None:
        bone_keys = ["rib", "vertebrae", "sternum", "scapula", "clavicula"]

    nii_files = sorted(ts_total_dir.glob("*.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError(f"No *.nii.gz found in: {ts_total_dir}")

    bone_mask = None
    ref_img = None
    used = []

    for f in nii_files:
        name = f.name.replace(".nii.gz", "")
        if any(k in name for k in bone_keys):
            m, img = load_mask_nii(f)

            if bone_mask is None:
                bone_mask = m.copy()
                ref_img = img
            else:
                if m.shape != bone_mask.shape:
                    raise ValueError(f"Shape mismatch: {f.name}={m.shape}, bone={bone_mask.shape}")
                bone_mask |= m

            used.append(f.name)

    if bone_mask is None or ref_img is None:
        raise RuntimeError("No matched bone masks found. Check bone_keys or TotalSeg output.")

    '''
    print("\n[Bone merge] used files (first 30):")
    for u in used[:30]:
        print("  -", u)
    if len(used) > 30:
        print(f"  ... ({len(used)} files total)")
    '''

    # fill holes in each 2D slice (optional)
    if fill_holes_2d:
        bone_filled = np.zeros_like(bone_mask, dtype=bool)
        for k in range(bone_mask.shape[-1]):
            bone_filled[..., k] = binary_fill_holes(bone_mask[..., k])
        bone_mask = bone_filled

    # save union mask
    save_mask_like(ref_img, bone_mask, out_path)



# =========================
# Integrate bone mask with image
# =========================
def integrate_bone_mask(dicom_dir: Path, bone_out: Path):
    '''
    We will combine bone mask with the ct slice and return them in a list
    '''
    dcm_files = sort_dicom_slices(dicom_dir)
    Z = len(dcm_files)

    ct_temp = extract_data_thr(dcm_files[0])
    H, W = ct_temp.shape
    ct_bone_zyx = np.empty((Z, H, W), dtype = np.int16)
    ct_body_zyx = np.empty((Z, H, W), dtype = np.int16)

    mask_zyx = load_bone_mask_as_zyx(bone_out, (Z, H, W))
    mask_zyx = mask_zyx[::-1]
    mask_zyx = np.rot90(mask_zyx, k = 1, axes = (1,2))

    print("Masking...")

    for i, dcm_file in enumerate(dcm_files):
        ct = extract_data_thr(dcm_file)

        bone_dila = binary_dilation(mask_zyx[i], iterations = 1)
        bone_eros_mask = binary_erosion(bone_dila, iterations = 1) # bone_eros_mask is bool
        bone_eros_mask = bone_eros_mask.astype(np.uint8) # turn into 0/1

        ct_bone = ct.copy()
        ct_body = ct.copy()
        ct_body = remove_couch(ct_body)
        ct_bone = ct_bone * bone_eros_mask
        ct_bone_zyx[i] = ct_bone
        ct_body_zyx[i] = ct_body - ct_bone

    print("FINISHED Masking")
    return ct_body_zyx, ct_bone_zyx


# =========================
# Conduct the beam projection
# =========================
def beam_projection(mu_total_zyx, spacing_cm):
    """
    Parallel-beam projection (line integral).
    
    mu_total_zyx: (Z, Y, X) linear attenuation volume, unit 1/cm
    spacing_cm: (dz, dy, dx) in cm
    
    Returns:
      P: line integral (2D)
      I: intensity (optional), I = I0 * exp(-P)
    """
    if mu_total_zyx.ndim != 3:
        raise ValueError(f"mu_total_zyx must be 3D (Z,Y,X). Got {mu_total_zyx.shape}")

    dz, dy, dx = spacing_cm
    mu = mu_total_zyx.astype(np.float32, copy=False)
    mu = np.clip(mu, 0.0, None)
    P = mu.sum(axis=1) * dy     # (Z, X)

    return P



# =========================
# Conduct the pathlength projection
# =========================
def pathlength_proj(soft_bone, spacing_cm):
    dz, dy, dx = spacing_cm
    mask = (soft_bone > 0.0).astype(np.float32)
    L = mask.sum(axis=1) * dy
    return L


# =========================
# Get mu for projection
# =========================
def component_to_mu(ct_component_zyx, mu_value):
    """
    Turn "body" to mu
    
    ct_component_zyx: (Z,Y,X)
    mu_value: constant mu (1/cm)
    thr: threshold to decide material region
    """
    comp = ct_component_zyx.astype(np.float32, copy=False)
    mask = comp > 0.0
    mu = np.zeros_like(comp, dtype=np.float32)
    mu[mask] = float(mu_value)
    return mu


# =========================
# Get high energy projection
# =========================
def get_high_energy_proj(ct_body_zyx, ct_bone_zyx, spacing_cm):

    mu_body = component_to_mu(ct_body_zyx, tissue_value_high)
    mu_bone = component_to_mu(ct_bone_zyx, bone_value_high)
    mu_total = mu_body + mu_bone
    return beam_projection(mu_total, spacing_cm)


# =========================
# Get low energy projection
# =========================
def get_low_energy_proj(ct_body_zyx, ct_bone_zyx, spacing_cm):
    mu_body = component_to_mu(ct_body_zyx, tissue_value_low)
    mu_bone = component_to_mu(ct_bone_zyx, bone_value_low)
    mu_total = mu_body + mu_bone
    return beam_projection(mu_total, spacing_cm)


# =========================
# Get pathlength projection
# =========================
def get_pathlength_proj(ct_body_zyx, ct_bone_zyx, spacing_cm):
    soft_bone = ct_body_zyx + ct_bone_zyx
    return pathlength_proj(soft_bone, spacing_cm)


# =========================
# Stretch to match the resolution
# =========================
def stretch_img(img2d):
    """
    Resample 2D image to 512x512 (or out_hw).
    order=1: bilinear (recommended for projection images)
    """
    order = 1
    H, W = img2d.shape
    zy = out_h / H
    zx = out_w / W
    return zoom(img2d.astype(np.float32, copy = False), (zy, zx), order = order)


# =========================
# Get spacing to conduct projection
# =========================
def get_spacing(dicom_dir: Path):
    dcm_paths = sorted(Path(dicom_dir).glob("*.dcm"))
    if len(dcm_paths) < 2:
        raise ValueError("Need at least 2 DICOM slices to compute dz reliably.")

    dsets = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in dcm_paths]

    dy_mm, dx_mm = map(float, dsets[0].PixelSpacing)

    z = np.array([float(ds.ImagePositionPatient[2]) for ds in dsets], dtype=np.float32)
    z_sorted = np.sort(z)
    dz_mm = float(np.median(np.abs(np.diff(z_sorted))))

    return (dz_mm/10, dy_mm/10, dx_mm/10)
    

# =========================
# Run the projection
# =========================
def proj_img_main(dicom_dir, ct_body_zyx, ct_bone_zyx):
    """
    End-to-end demo:
      1) compute spacing (cm)
      2) build ct_body_zyx / ct_bone_zyx from CT + TotalSeg bone mask
      3) compute low/high projections P
      4) optionally convert to intensity I = exp(-P)
      5) plot results
    """
    # Get the space
    spacing_cm = get_spacing(dicom_dir)   # (dz, dy, dx) in cm

    # Get the projection
    P_low_y  = get_low_energy_proj(ct_body_zyx, ct_bone_zyx, spacing_cm)
    P_high_y = get_high_energy_proj(ct_body_zyx, ct_bone_zyx, spacing_cm)
    Pathlength_y = get_pathlength_proj(ct_body_zyx, ct_bone_zyx, spacing_cm)
    
    P_low_y_stretched = stretch_img(P_low_y)
    P_high_y_stretched = stretch_img(P_high_y)
    Pathlength_y_stretched = stretch_img(Pathlength_y)
    
    return P_low_y_stretched, P_high_y_stretched, Pathlength_y_stretched

    # return arrays for debugging if you want
    #return P_low, P_high, I_low, I_high


# =========================
# Get individual projection
# =========================
def ct_to_proj(work_dir, dicom_dir, if_fast):
    ts_total_dir = work_dir / "ts_total"
    bone_out = work_dir / "bone_mask.nii.gz"

    try:
        # start clean
        safe_cleanup(work_dir)

        work_dir.mkdir(parents=True, exist_ok=True)
        get_bone_mask(dicom_dir, ts_total_dir, if_fast, bone_out)
            
        ct_body_zyx, ct_bone_zyx = integrate_bone_mask(dicom_dir, bone_out)
            
        low_energy_proj, high_energy_proj, pathlength_proj = proj_img_main(dicom_dir, ct_body_zyx, ct_bone_zyx)
        
        return low_energy_proj, high_energy_proj, pathlength_proj

    finally:
        # this CT ends -> delete ALL
        safe_cleanup(work_dir)
        

def main():

    output_dir_hi.mkdir(parents=True, exist_ok=True)
    output_dir_low.mkdir(parents=True, exist_ok=True)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in main_dir.iterdir()
                           if p.is_dir() and p.name.startswith("LIDC-IDRI-")])

    for i, dicom_dir in enumerate(patient_dirs, 1):
        pid = dicom_dir.name  # e.g. LIDC-IDRI-0001

        out_hi = output_dir_hi / f"{pid}-hi.npy"
        out_low = output_dir_low / f"{pid}-low.npy"
        out_path = output_dir_path / f"{pid}-path.npy"

        dcm_count = len(list(dicom_dir.glob("*.dcm")))
        if dcm_count == 0:
            print(f"[{i}/{len(patient_dirs)}] {pid}: no *.dcm -> skip")
            continue

        print(f"\n[{i}/{len(patient_dirs)}] Processing {pid}  (dcm={dcm_count})")

        try:
            low_proj, high_proj, path_proj = ct_to_proj(work_dir, dicom_dir, if_fast)

            np.save(out_hi, high_proj.astype(np.float32))
            np.save(out_low, low_proj.astype(np.float32))
            np.save(out_path, path_proj.astype(np.float32))

        except Exception as e:
            print(f"[FAIL] {pid}: {e}")

if __name__ == "__main__":
    main()
