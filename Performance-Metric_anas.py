#!/usr/bin/env python3
"""
Performance Metrics Evaluation Script for HierAdaptMR / PromptMR+
Computes SSIM and NMSE between reconstructed images and ground truth.
Outputs results to an Excel file with metadata.

Usage:
    python Performance-Metric_anas.py \
        --recon_dir /path/to/reconstructed/files \
        --gt_dir /path/to/ground_truth/files \
        --output_dir /path/to/output/excel \
        --acc_factors 8 16 24
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from typing import Dict, Tuple, List

from mri_utils.utils import ssim, nmse, loadmat


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MRI reconstruction performance")
    parser.add_argument("--recon_dir", type=str, required=True,
                        help="Directory containing reconstructed .mat files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth .mat files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output Excel file")
    parser.add_argument("--acc_factors", type=int, nargs="+", default=[8, 16, 24],
                        help="Acceleration factors to evaluate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for evaluation")
    return parser.parse_args()


def extract_metadata_from_path(filepath: str, base_dir: str) -> Dict[str, str]:
    """
    Extract metadata (Center, Contrast, Patient ID, etc.) from file path.
    
    Expected structure:
    .../Center001/UIH_30T_umr780/P007/cine_lax_3ch.mat
    """
    rel_path = filepath.replace(str(base_dir) + '/', '')
    parts = rel_path.split('/')
    
    metadata = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'center': '',
        'scanner': '',
        'patient_id': '',
        'contrast': '',
        'slice_name': ''
    }
    
    if len(parts) >= 5:
        metadata['center'] = parts[0]  # e.g., Center001
        metadata['scanner'] = parts[1]  # e.g., UIH_30T_umr780
        metadata['patient_id'] = parts[2]  # e.g., P007
        metadata['contrast'] = parts[3]  # e.g., cine_lax_3ch
        metadata['slice_name'] = parts[4]  # e.g., cine_lax_3ch.mat
    
    return metadata


def load_reconstruction(filepath: str) -> np.ndarray:
    """Load reconstruction from .mat file."""
    try:
        data = loadmat(filepath)
        # Look for common keys in reconstruction files
        for key in ['reconstruction', 'img4ranking', 'recons', 'output']:
            if key in data:
                return data[key]
        # If no standard key found, return first array
        for key in data.keys():
            return data[key]
        raise KeyError(f"No valid reconstruction data found in {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_ground_truth(filepath: str) -> np.ndarray:
    """Load ground truth from .mat file."""
    try:
        data = loadmat(filepath)
        # Look for common keys in ground truth files
        for key in ['kspace', 'image', 'img', 'data', 'img4ranking']:
            if key in data:
                gt_data = data[key]
                # If kspace, convert to image domain
                if key == 'kspace':
                    from mri_utils import ifft2c, rss_complex
                    if gt_data.ndim == 5:  # (t, z, nc, y, x) or similar
                        # Combine coils using RSS
                        gt_data = rss_complex(gt_data)
                    elif gt_data.ndim == 4:
                        gt_data = rss_complex(gt_data)
                return gt_data
        # If no standard key found, return first array
        for key in data.keys():
            return data[key]
        raise KeyError(f"No valid ground truth data found in {filepath}")
    except Exception as e:
        print(f"Error loading ground truth {filepath}: {e}")
        return None


def compute_metrics(args: Tuple[str, str, Dict]) -> Dict:
    """
    Compute SSIM and NMSE for a single file pair.
    
    Args:
        args: Tuple of (recon_path, gt_path, metadata)
    
    Returns:
        Dictionary with metrics and metadata
    """
    recon_path, gt_path, metadata = args
    
    result = {
        **metadata,
        'ssim': np.nan,
        'nmse': np.nan,
        'error': ''
    }
    
    try:
        # Load reconstruction
        recon = load_reconstruction(recon_path)
        if recon is None:
            result['error'] = f"Failed to load reconstruction: {recon_path}"
            return result
        
        # Load ground truth
        gt = load_ground_truth(gt_path)
        if gt is None:
            result['error'] = f"Failed to load ground truth: {gt_path}"
            return result
        
        # Ensure both are numpy arrays
        recon = np.array(recon)
        gt = np.array(gt)
        
        # Handle complex data - take magnitude
        if np.iscomplexobj(recon):
            recon = np.abs(recon)
        if np.iscomplexobj(gt):
            gt = np.abs(gt)
        
        # Normalize to [0, 1] range for SSIM
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
        
        # Ensure same shape - may need to handle temporal/slice dimensions
        if recon.shape != gt.shape:
            # Try to match shapes by taking mean or selecting appropriate dimension
            if recon.ndim > gt.ndim:
                if recon.ndim == 4 and gt.ndim == 3:
                    recon = recon.mean(axis=0)  # Average over time
                elif recon.ndim == 5:
                    recon = recon.mean(axis=(0, 1))  # Average over time and slices
            elif gt.ndim > recon.ndim:
                if gt.ndim == 4 and recon.ndim == 3:
                    gt = gt.mean(axis=0)
                elif gt.ndim == 5:
                    gt = gt.mean(axis=(0, 1))
            
            # Crop or pad if still different
            min_shape = tuple(min(a, b) for a, b in zip(recon.shape, gt.shape))
            if recon.shape != min_shape:
                slices = tuple(slice(0, s) for s in min_shape)
                recon = recon[slices]
            if gt.shape != min_shape:
                slices = tuple(slice(0, s) for s in min_shape)
                gt = gt[slices]
        
        # Ensure 3D for SSIM (slices, height, width)
        if recon.ndim == 2:
            recon = recon[np.newaxis, ...]
            gt = gt[np.newaxis, ...]
        elif recon.ndim > 3:
            # Flatten extra dimensions into slice dimension
            orig_shape = recon.shape
            recon = recon.reshape(-1, *orig_shape[-2:])
            gt = gt.reshape(-1, *gt.shape[-2:])
        
        # Compute SSIM
        ssim_val = ssim(gt_norm, recon_norm)
        result['ssim'] = float(ssim_val) if isinstance(ssim_val, np.ndarray) else ssim_val
        
        # Compute NMSE
        nmse_val = nmse(gt_norm, recon_norm)
        result['nmse'] = float(nmse_val) if isinstance(nmse_val, np.ndarray) else nmse_val
        
    except Exception as e:
        result['error'] = str(e)
        print(f"Error processing {recon_path}: {e}")
    
    return result


def find_matching_files(recon_dir: str, gt_dir: str, acc_factor: int) -> List[Tuple[str, str]]:
    """
    Find matching reconstruction and ground truth files.
    
    Assumes reconstructions are organized by acceleration factor:
    recon_dir/
        acc8/
            Center001/.../file.mat
        acc16/
            Center001/.../file.mat
        acc24/
            Center001/.../file.mat
    
    Ground truth is in gt_dir with same relative structure.
    """
    pairs = []
    recon_path = Path(recon_dir) / f"acc{acc_factor}"
    
    if not recon_path.exists():
        # Try without acc subdirectory
        recon_path = Path(recon_dir)
    
    gt_path = Path(gt_dir)
    
    # Find all .mat files in recon directory
    for recon_file in recon_path.glob('**/*.mat'):
        # Construct corresponding GT path
        rel_path = recon_file.relative_to(recon_path)
        gt_file = gt_path / rel_path
        
        if gt_file.exists():
            pairs.append((str(recon_file), str(gt_file)))
        else:
            # Try to find any matching .mat file in GT directory
            gt_files = list(gt_path.glob(f'**/{rel_path.name}'))
            if gt_files:
                pairs.append((str(recon_file), str(gt_files[0])))
    
    return pairs


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    print(f"Evaluating reconstructions in: {args.recon_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Acceleration factors: {args.acc_factors}")
    print("-" * 60)
    
    for acc_factor in args.acc_factors:
        print(f"\nProcessing acceleration factor: {acc_factor}x")
        
        # Find matching file pairs
        file_pairs = find_matching_files(args.recon_dir, args.gt_dir, acc_factor)
        
        if not file_pairs:
            print(f"  No matching files found for acc={acc_factor}")
            continue
        
        print(f"  Found {len(file_pairs)} file pairs")
        
        # Prepare arguments for parallel processing
        eval_args = []
        for recon_path, gt_path in file_pairs:
            metadata = extract_metadata_from_path(recon_path, args.recon_dir)
            metadata['acc_factor'] = acc_factor
            eval_args.append((recon_path, gt_path, metadata))
        
        # Process in parallel
        print(f"  Computing metrics with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(compute_metrics, eval_args)
        
        all_results.extend(results)
        
        # Print summary
        ssim_vals = [r['ssim'] for r in results if not np.isnan(r['ssim'])]
        nmse_vals = [r['nmse'] for r in results if not np.isnan(r['nmse'])]
        
        if ssim_vals:
            print(f"  SSIM - Mean: {np.mean(ssim_vals):.4f}, Std: {np.std(ssim_vals):.4f}")
        if nmse_vals:
            print(f"  NMSE - Mean: {np.mean(nmse_vals):.4f}, Std: {np.std(nmse_vals):.4f}")
    
    # Create DataFrame and save to Excel
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Reorder columns
        cols = ['acc_factor', 'center', 'scanner', 'patient_id', 'contrast', 
                'filename', 'ssim', 'nmse', 'error', 'filepath']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols + [c for c in df.columns if c not in available_cols]]
        
        # Save to Excel
        excel_path = output_dir / "performance_metrics.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"\n✓ Results saved to: {excel_path}")
        
        # Also save as CSV
        csv_path = output_dir / "performance_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved to: {csv_path}")
        
        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        for acc in args.acc_factors:
            acc_df = df[df['acc_factor'] == acc]
            ssim_mean = acc_df['ssim'].mean()
            nmse_mean = acc_df['nmse'].mean()
            
            if not np.isnan(ssim_mean):
                print(f"Accel {acc}x: SSIM = {ssim_mean:.4f} ± {acc_df['ssim'].std():.4f}, "
                      f"NMSE = {nmse_mean:.4f} ± {acc_df['nmse'].std():.4f}")
        
        print("=" * 60)
    else:
        print("\n⚠ No results to save. Check input directories and file paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
