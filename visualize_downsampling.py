import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from diffusion.aliasing_diffusion import AliasingDiffusion

def main():
    # Load image
    img_path = r"data\train\synthetic\polInt1_polOut2_tomRawAntSeg.npy"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return
        
    # Load the npy array
    # The shape is expected to be (B-scans, Z, X, 2)
    data = np.load(img_path)
    
    # Select middle B-scan
    bscan_idx = data.shape[0] // 2
    bscan = data[bscan_idx, :, :, :]  # Shape: (Z, X, 2)
    
    # Transpose to (2, Z, X) for the channels (real, imag)
    bscan_transposed = np.transpose(bscan, (2, 0, 1))
    
    # Add batch dimension and convert to tensor
    # Expected shape: [Batch, 2, Z, X]
    x_start = torch.from_numpy(bscan_transposed).unsqueeze(0).float()
    Z, X = x_start.shape[2], x_start.shape[3]
    
    # Create diffusion model
    dummy_denoise = nn.Identity()
    timesteps = 1000
    diffusion = AliasingDiffusion(
        denoise_fn=dummy_denoise,
        image_size=X,
        timesteps=timesteps,
        target_subsample_factor=16
    )
    
    # Timesteps to visualize
    ts = [0, 100, 300, 500, 1000]
    
    fig, axes = plt.subplots(1, len(ts), figsize=(15, 3))
    
    for i, t_val in enumerate(ts):
        t = torch.tensor([t_val])
        
        # Apply downsampling (forward diffusion)
        x_blur = diffusion.q_sample(x_start, t)
        
        # Calculate magnitude
        real = x_blur[0, 0, :, :].numpy()
        imag = x_blur[0, 1, :, :].numpy()
        magnitude = np.sqrt(real**2 + imag**2)
        
        # Convert magnitude to dB scale for typical OCT visualization
        magnitude_db = 20 * np.log10(magnitude + 1e-8)
        
        ax = axes[i]
        ax.imshow(magnitude_db, cmap='gray', vmin=np.percentile(magnitude_db, 1), vmax=np.percentile(magnitude_db, 99))
        ax.set_title(f"t = {t_val}")
        ax.axis('off')
        
    plt.tight_layout()
    out_path = "downsampling_visualization.png"
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
