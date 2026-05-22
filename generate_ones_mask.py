import os
import h5py
import numpy as np

# Adjust the root path if your bitlocker mount point differs
base_dir = "/media/sbme26-600k/bitlocker_mount/CardiacCreed/cmrxrecon2025/ChallengeData/MultiCoil/Cine/TrainingSet/Mask_TaskAll/Center005/Siemens_30T_Vida/P001/"
target_mask = os.path.join(base_dir, "cine_lax_mask_Uniform8.mat")
backup_mask = os.path.join(base_dir, "cine_lax_mask_Uniform8_BACKUP.mat")

def create_redundant_mask():
    if not os.path.exists(target_mask) and not os.path.exists(backup_mask):
        print(f"Error: Target mask not found at {base_dir}")
        return

    # 1. Backup the original mask to prevent data loss
    if not os.path.exists(backup_mask):
        os.rename(target_mask, backup_mask)
        print(f"Backed up original mask to: {backup_mask}")
    else:
        print("Backup already exists. Reading from backup.")
    
    # 2. Read the exact shape and dataset key from the original file
    with h5py.File(backup_mask, 'r') as hf_in:
        keys = list(hf_in.keys())
        # CMRxRecon masks usually use 'mask' or 'mask4ranking' as the key
        mask_key = 'mask' if 'mask' in keys else keys[0]
        original_shape = hf_in[mask_key].shape
        original_dtype = hf_in[mask_key].dtype
        
        print(f"Original shape: {original_shape}, Target key: '{mask_key}'")
        
        # 3. Generate a mask of all ones
        ones_mask = np.ones(original_shape, dtype=original_dtype)
        
        # 4. Save the redundant mask as a v7.3 HDF5 file
        with h5py.File(target_mask, 'w') as hf_out:
            hf_out.create_dataset(mask_key, data=ones_mask)
            
    print(f"Successfully generated fully sampled HDF5 mask at: {target_mask}")

if __name__ == "__main__":
    create_redundant_mask()