import torch
import torch.nn as nn
import torch.fft

class SpectralColdDiffusion(nn.Module):
    """
    Implements Cold Diffusion for CS-OCT using 1D Lateral Spectral Aliasing.
    Degrades images by progressively filtering out high lateral frequencies.
    """
    def __init__(self, denoise_fn: nn.Module, timesteps: int = 100):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.num_timesteps = timesteps

    def _get_bandpass_mask(self, W: int, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Creates a 1D bandpass mask for the lateral frequencies.
        At t=0 (clean), it keeps 100% of the frequencies.
        At t=timesteps (fully degraded), it keeps a small central fraction.
        """
        B = t.shape[0]
        center = W // 2
        
        # Calculate the ratio of frequencies to keep. 
        # e.g., keeping from 100% down to 10% of the spectrum
        min_keep_ratio = 0.1 
        keep_ratio = 1.0 - ((1.0 - min_keep_ratio) * (t / self.num_timesteps))
        
        # Calculate cutoff indices
        cutoffs = (center * keep_ratio).long()
        
        mask = torch.zeros((B, 1, 1, W), device=device)
        for i in range(B):
            c = cutoffs[i].item()
            if c > 0:
                mask[i, :, :, center - c : center + c] = 1.0
                
        return mask

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        The Forward Process: Applies the sub-Nyquist degradation.
        Expects x0 as a 2-channel Real/Imaginary tensor: (B, 2, H, W)
        """
        # Convert 2-channel (Re, Im) back to complex64 for FFT
        x0_complex = torch.complex(x0[:, 0:1, ...], x0[:, 1:2, ...])
        B, _, H, W = x0_complex.shape

        # 1D FFT along the lateral axis (Width)
        x_freq = torch.fft.fftshift(torch.fft.fft(x0_complex, dim=-1, norm='ortho'), dim=-1)
        
        # Get mask and apply it
        mask = self._get_bandpass_mask(W, t, x0.device)
        x_t_freq = x_freq * mask
        
        # Inverse FFT back to spatial domain
        x_t_spatial = torch.fft.ifft(torch.fft.ifftshift(x_t_freq, dim=-1), dim=-1, norm='ortho')
        
        # Convert back to 2-channel Real/Imaginary for the U-Net
        x_t_out = torch.cat([x_t_spatial.real, x_t_spatial.imag], dim=1)
        return x_t_out, mask

    def frequency_data_consistency(self, x_pred: torch.Tensor, x_measured: torch.Tensor, mask: torch.Tensor):
        """
        The DC Step: Replaces the network's predicted low frequencies with the 
        true, measured low frequencies to perfectly preserve phase.
        """
        # Convert to complex
        pred_complex = torch.complex(x_pred[:, 0:1, ...], x_pred[:, 1:2, ...])
        meas_complex = torch.complex(x_measured[:, 0:1, ...], x_measured[:, 1:2, ...])

        # Transform both to frequency domain
        pred_freq = torch.fft.fftshift(torch.fft.fft(pred_complex, dim=-1, norm='ortho'), dim=-1)
        meas_freq = torch.fft.fftshift(torch.fft.fft(meas_complex, dim=-1, norm='ortho'), dim=-1)

        # Overwrite the predicted frequencies with measured frequencies where mask == 1
        # Where mask == 0, keep the network's hallucinated high frequencies
        corrected_freq = (pred_freq * (1 - mask)) + (meas_freq * mask)

        # Inverse transform
        corrected_spatial = torch.fft.ifft(torch.fft.ifftshift(corrected_freq, dim=-1), dim=-1, norm='ortho')
        
        return torch.cat([corrected_spatial.real, corrected_spatial.imag], dim=1)

    def forward(self, x0: torch.Tensor):
        """
        Training forward pass. Calculates the loss.
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample a random timestep for each image in the batch
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=device).long()
        
        # Degrade the image
        x_blur, mask = self.q_sample(x0, t)
        
        # Predict the clean image (x0) from the degraded image
        x_recon = self.denoise_fn(x_blur, t)
        
        # Custom Loss: L1 over the whole image.
        # NOTE: We DO NOT apply Data Consistency (DC) before calculating the loss during training!
        # If you apply DC here, the gradients for the low frequencies become zero, and the network
        # will learn to output garbage in the low frequencies because it knows DC will fix them.
        # The network must learn to reconstruct the full signal. DC is strictly an inference-time tool.
        loss = torch.nn.functional.l1_loss(x_recon, x0)
        
        return loss