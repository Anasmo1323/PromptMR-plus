#!/usr/bin/env python3
"""
MRI Reconstruction Evaluator for HierAdaptMR / PromptMR Plus
- Loads GT k-space (5D: T,Z,C,Y,X), applies IFFT+RSS → magnitude (T,Z,Y,X)
- Loads inference output (4D: T,Z,Y,X) from img4ranking
- Evaluates central Z slice across all time frames, averaged SSIM + NMSE
- Normalizes both GT and pred by per-volume max before comparison
- Outputs per-sample metrics with full metadata (Patient, Center, Vendor, Acc)
- Aggregates results by acceleration factor (8/16/24)
"""
import argparse
from pathlib import Path
import numpy as np
import h5py
import scipy.io
import re
from skimage.metrics import structural_similarity
import pandas as pd
import sys


def ifft2c_np(kspace: np.ndarray) -> np.ndarray:
    """Centered inverse FFT on the last two axes, with ortho normalization."""
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1), norm="ortho")
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def rss_complex_np(x: np.ndarray, coil_dim: int = 2) -> np.ndarray:
    """Root-sum-of-squares coil combination for complex arrays."""
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=coil_dim))


def load_inference_mat(path: Path) -> np.ndarray:
    """Load inference output from .mat file (handles both scipy.io and h5py formats)."""
    # Try h5py first (MATLAB v7.3)
    try:
        with h5py.File(str(path), 'r') as f:
            if "img4ranking" not in f:
                raise KeyError(f'Expected key "img4ranking" not found in {path}')
            img = f["img4ranking"][()]
    except Exception:
        # Fallback to scipy.io for older .mat files
        data = scipy.io.loadmat(str(path))
        if "img4ranking" not in data:
            raise KeyError(f'Expected key "img4ranking" not found in {path}')
        img = data["img4ranking"]
    
    if img.ndim != 4:
        raise ValueError(f"Expected img4ranking to be 4D (T,Z,Y,X), got {img.ndim}D for {path}")
    return np.asarray(img, dtype=np.float32)


def load_ground_truth_kspace(path: Path) -> np.ndarray:
    """Load GT k-space from .mat file, apply IFFT+RSS to get magnitude image (T,Z,Y,X)."""
    with h5py.File(str(path), "r") as hf:
        if "kspace" not in hf:
            raise KeyError(f'Expected key "kspace" not found in {path}')
        kspace = hf["kspace"][()]

    # Handle complex struct arrays (MATLAB v7.3 format)
    if hasattr(kspace.dtype, "names") and kspace.dtype.names == ("real", "imag"):
        kspace = kspace["real"] + 1j * kspace["imag"]

    # Ensure kspace is 5D: (T, Z, Coils, Y, X)
    if kspace.ndim == 4:
        # Assume (Z, Coils, Y, X) -> add time dim
        kspace = np.expand_dims(kspace, 0)
    
    if kspace.ndim != 5:
        raise ValueError(f"Expected ground truth kspace to be 5D (T,Z,C,Y,X), got {kspace.ndim}D")
    
    # Apply centered IFFT on last two axes (Y, X)
    img = ifft2c_np(kspace)  # Still complex, shape (T, Z, C, Y, X)
    
    # RSS coil combination along coil_dim=2
    combined = rss_complex_np(img, coil_dim=2)  # -> (T, Z, Y, X), magnitude
    
    return np.abs(combined).astype(np.float32)


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Compute SSIM (averaged over time) and NMSE on normalized 4D volumes (T,Y,X)."""
    if gt.shape != pred.shape:
        raise ValueError(f"GT shape {gt.shape} does not match pred shape {pred.shape}")
    
    # Normalize to [0,1] per volume using max
    gt_norm = gt / (gt.max() + 1e-11)
    pred_norm = pred / (pred.max() + 1e-11)
    
    t = gt_norm.shape[0]
    ssim_values = []
    data_range = 1.0  # After normalization
    
    for ii in range(t):
        ssim_values.append(
            structural_similarity(
                gt_norm[ii],
                pred_norm[ii],
                data_range=data_range,
                gaussian_weights=True,
                use_sample_covariance=False,
            )
        )
    ssim_avg = float(np.mean(ssim_values))
    nmse = float(np.linalg.norm(gt_norm - pred_norm) ** 2 / (np.linalg.norm(gt_norm) ** 2 + 1e-11))
    return ssim_avg, nmse


def extract_metadata(mat_path: Path, root: Path) -> dict:
    """Extract Patient, Center, Vendor, Acc from path structure."""
    parts = mat_path.relative_to(root).parts
    metadata = {
        "Patient": "Unknown",
        "Center": "Unknown", 
        "Vendor": "Unknown",
        "Acc": 8
    }
    
    # Extract acceleration from parent directory (acc_8, acc_16, acc_24)
    acc_match = re.match(r'acc_(\d+)', mat_path.parent.name)
    if acc_match:
        metadata["Acc"] = int(acc_match.group(1))
    
    # Find Patient (P###) and Center (CenterXXX) in path
    for part in parts:
        if re.match(r'^P\d+$', part):
            metadata["Patient"] = part
        elif part.startswith('Center'):
            metadata["Center"] = part
        elif part.lower() in ['siemens', 'ge', 'philips', 'canon']:
            metadata["Vendor"] = part.capitalize()
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Compare inference outputs to ground truth k-space data.")
    parser.add_argument("--inference_root", type=Path, required=True, help="Root directory of inference .mat files")
    parser.add_argument("--ground_truth_root", type=Path, required=True, help="Root directory of ground truth .mat files")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional output CSV for per-file metrics")
    parser.add_argument("--output_excel", type=Path, default=None, help="Optional output Excel for aggregated results")
    args = parser.parse_args()

    root = args.inference_root
    gtruth_root = args.ground_truth_root
    all_results = []
    missing = []
    errors = []

    # Open CSV for live writing if specified
    csvfile = None
    writer = None
    if args.output_csv:
        import csv
        csvfile = open(args.output_csv, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["path", "patient", "center", "vendor", "acc", "ssim_avg", "nmse"])
        csvfile.flush()

    try:
        # Group inference files by base filename (ignoring acc_X subdirectory)
        file_groups = {}
        for mat_path in sorted(root.rglob("*.mat")):
            rel = mat_path.relative_to(root)
            rel_parts = rel.parts
            
            # Extract base path by removing acc_X directory
            if rel_parts[0].startswith("acc_"):
                base_rel = Path(*rel_parts[1:])
            else:
                base_rel = rel
            
            base_key = str(base_rel)
            if base_key not in file_groups:
                file_groups[base_key] = []
            file_groups[base_key].append(mat_path)
        
        # Process each unique file with all 3 acceleration factors
        for base_key, mat_paths in file_groups.items():
            # Ensure we have all 3 acc factors, or process what's available
            for mat_path in mat_paths:
                rel = mat_path.relative_to(root)
                
                # Extract metadata
                meta = extract_metadata(mat_path, root)
                
                # Construct GT path: remove acc_X from relative path
                rel_parts = rel.parts
                if rel_parts[0].startswith("acc_"):
                    gt_rel = Path(*rel_parts[1:])
                else:
                    gt_rel = rel
                gt_path = gtruth_root / gt_rel
                
                if not gt_path.exists():
                    missing.append(str(mat_path.relative_to(root)))
                    continue

                try:
                    # Load inference output: (T, Z, Y, X) magnitude
                    inf_img = load_inference_mat(mat_path)
                    
                    # Load GT kspace and convert to magnitude image: (T, Z, Y, X)
                    gt_combined = load_ground_truth_kspace(gt_path)
                    
                    # Check T and Z match
                    if inf_img.shape[0] != gt_combined.shape[0] or inf_img.shape[1] != gt_combined.shape[1]:
                        raise ValueError(
                            f"Shape mismatch: inference {inf_img.shape} vs ground truth {gt_combined.shape} for {rel}"
                        )
                    
                    # Select central Z slice: (T, Y, X)
                    z_center = gt_combined.shape[1] // 2
                    gt_central = gt_combined[:, z_center, :, :]  # (T, Y, X)
                    pred_central = inf_img[:, z_center, :, :]     # (T, Y, X)
                    
                    # Handle possible transpose if dimensions are swapped
                    if gt_central.shape != pred_central.shape:
                        if gt_central.shape == (pred_central.shape[0], pred_central.shape[2], pred_central.shape[1]):
                            pred_central = pred_central.transpose(0, 2, 1)
                        else:
                            raise ValueError(f"GT shape {gt_central.shape} does not match pred shape {pred_central.shape}")
                    
                    # Compute metrics on central slice across all time frames
                    ssim_avg, nmse_val = compute_metrics(gt_central, pred_central)
                    
                    result = {
                        "path": str(rel),
                        "patient": meta["Patient"],
                        "center": meta["Center"],
                        "vendor": meta["Vendor"],
                        "acc": meta["Acc"],
                        "ssim_avg": ssim_avg,
                        "nmse": nmse_val
                    }
                    all_results.append(result)
                    
                    if writer:
                        writer.writerow([
                            result["path"], result["patient"], result["center"], 
                            result["vendor"], result["acc"], result["ssim_avg"], result["nmse"]
                        ])
                        csvfile.flush()
                        
                except Exception as exc:
                    errors.append((str(rel), repr(exc)))
                    
    finally:
        if csvfile:
            csvfile.close()

    # Print summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"✅ File count: {len(all_results)}")
        print(f"📈 Average SSIM: {df['ssim_avg'].mean():.4f}")
        print(f"📉 Average NMSE: {df['nmse'].mean():.6f}")
        print()
        
        # Aggregate by acceleration factor
        print("📊 Metrics by Acceleration:")
        agg = df.groupby("acc")[["ssim_avg", "nmse"]].agg(["mean", "std"]).round(4)
        print(agg)
        print()
        
        # Per-center breakdown if multiple centers exist
        if df["center"].nunique() > 1:
            print("🏥 Metrics by Center:")
            center_agg = df.groupby("center")[["ssim_avg", "nmse"]].mean().round(4)
            print(center_agg)
            print()
        
        # Top 10 by SSIM
        print("🏆 Top 10 by SSIM:")
        for row in df.nlargest(10, "ssim_avg").itertuples(index=False):
            print(f"  {row.path}: SSIM={row.ssim_avg:.4f}, NMSE={row.nmse:.6f}")
        
        # Worst 5 by SSIM
        print("\n⚠️  Worst 5 by SSIM:")
        for row in df.nsmallest(5, "ssim_avg").itertuples(index=False):
            print(f"  {row.path}: SSIM={row.ssim_avg:.4f}, NMSE={row.nmse:.6f}")
        
        # Save Excel if requested
        if args.output_excel:
            args.output_excel.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(args.output_excel, index=False)
            print(f"\n💾 Full results saved to: {args.output_excel}")
            
            # Also save aggregated summary
            summary_path = args.output_excel.with_name(args.output_excel.stem + "_summary" + args.output_excel.suffix)
            with pd.ExcelWriter(summary_path) as writer:
                df.to_excel(writer, sheet_name="Per-Sample", index=False)
                df.groupby("acc")[["ssim_avg", "nmse"]].agg(["mean", "std"]).round(4).to_excel(writer, sheet_name="By-Acc")
                if df["center"].nunique() > 1:
                    df.groupby("center")[["ssim_avg", "nmse"]].mean().round(4).to_excel(writer, sheet_name="By-Center")
            print(f"💾 Summary sheet saved to: {summary_path}")
            
    else:
        print("❌ No matching inference/ground truth file pairs were processed.")

    if missing:
        print(f"\n⚠️  Missing ground truth files for {len(missing)} inference files.")
        if len(missing) <= 10:
            for m in missing:
                print(f"  - {m}")
    if errors:
        print(f"\n❌ Errors for {len(errors)} files:")
        for path, err in errors[:20]:
            print(f"  {path}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())