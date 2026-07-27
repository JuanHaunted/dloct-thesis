import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class AliasingDiffusion(nn.Module):
    """
    Cold Diffusion engine for sub-Nyquist OCT complex field reconstruction.
    
    Args:
        denoise_fn (nn.Module): The neural network used for denoising during the reverse
            diffusion process. It should take a noisy image and a timestep as input and
            output a denoised image.
        image_size (int): The height and width of the square input images.
        timesteps (int, optional): The number of diffusion steps. Default is 1000
        target_subsample_factor (int, optional): The factor by which the input images are
            subsampled. For example, a factor of 4 means the input images are 4 times smaller
            in each dimension than the original Nyquist-sampled images. Default is 4. 
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        timesteps: int = 1000,
        target_subsample_factor: int = 4,
    ):
        """
        Cold Diffusion engine for sub-Nyquist OCT complex field reconstruction.
        """
        super().__init__()
        self.denoise_fn = denoise_fn 
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.target_subsample_factor = target_subsample_factor

    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Converts [Batch, 2, Z, X] Real tensor to [Batch, Z, X] Complex tensor."""
        return torch.complex(x[:, 0, ...], x[:, 1, ...])

    def _to_real(self, x: torch.Tensor) -> torch.Tensor:
        """Converts [Batch, Z, X] Complex tensor to [Batch, 2, Z, X] Real tensor."""
        return torch.stack([x.real, x.imag], dim=1)


    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The deterministic forward degradation process D(x, t).
        Args:
            x_start: Ground truth clean tomogram [Batch, 2, Z, X]
            t: Tensor of timesteps for the batch [Batch]
        """
        B, C, Z, X = x_start.shape
        device = x_start.device
        
        # 1. Move to complex physical space
        x_complex = self._to_complex(x_start)
        
        # 2. Transform to lateral spatial frequencies
        k_space = torch.fft.fft(x_complex, dim=-1)
        k_space_shifted = torch.fft.fftshift(k_space, dim=-1)

        # 3. Calculate time-dependent Gaussian masks for the entire batch
        min_ratio = 1.0 / self.target_subsample_factor 
        # t is [B], so current_ratio is [B]
        current_ratio = 1.0 - (t.float() / self.num_timesteps) * (1.0 - min_ratio)
        
        # sigma shape: [B, 1, 1] for broadcasting
        sigma = ((X * current_ratio) / 3.0).view(B, 1, 1)
        sigma = torch.clamp(sigma, min=1e-3) 

        # Create spatial coordinates
        coords = torch.arange(X, device=device, dtype=torch.float32).view(1, 1, X)
        center = X / 2.0
        
        # Gaussian formula: exp(- (x - mu)^2 / (2 * sigma^2))
        mask = torch.exp(-0.5 * ((coords - center) / sigma)**2)

        # 4. Apply mask and inverse transform
        k_space_degraded = k_space_shifted * mask
        k_space_unshifted = torch.fft.ifftshift(k_space_degraded, dim=-1)
        x_t_complex = torch.fft.ifft(k_space_unshifted, dim=-1)

        # 5. Return to UNet-compatible real tensor
        return self._to_real(x_t_complex)
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, lambda_phase: float = 1.0) -> torch.Tensor:
        """
        Calculates the training loss with explicit phase-wrapping constraints.
        """
        # 1. Degrade the image deterministically to timestep t
        x_blur = self.q_sample(x_start, t)
        
        # 2. The UNet predicts the clean ground truth x_0
        x_recon = self.denoise_fn(x_blur, t)
        
        # 3. Structural L1 Loss (Calculated on Real and Imaginary channels)
        # This handles the baseline amplitude and structural constraints.
        loss_l1 = F.l1_loss(x_start, x_recon)
        
        # 4. Phase-Aware Loss
        # In our tensor, Channel 0 is Real, Channel 1 is Imaginary
        theta_true = torch.atan2(x_start[:, 1, ...], x_start[:, 0, ...])
        theta_pred = torch.atan2(x_recon[:, 1, ...], x_recon[:, 0, ...])
        
        # 1 - cos(delta_theta) ensures that an error of exactly 2*pi yields a loss of 0
        loss_phase = torch.mean(1.0 - torch.cos(theta_pred - theta_true))
        
        # 5. Combined Total Loss
        # Here lambda phase is a constant to level the loses
        total_loss = loss_l1 + lambda_phase * loss_phase
        
        # (Optional) You might want to log loss_l1 and loss_phase separately 
        # to your tracker (Weights & Biases, Comet, etc.) to watch them converge.
        return total_loss
    

    @torch.no_grad()
    def sample(self, x_T: torch.Tensor, enforce_dc: bool = False) -> torch.Tensor:
        """
        Iteratively reconstructs the dense OCT scan from the sub-Nyquist scan.
        Args:
            x_T: The raw, sub-sampled measurement [Batch, 2, Z, X]
            enforce_dc: Whether to explicitly enforce Data Consistency in k-space.
        """
        self.denoise_fn.eval()
        
        b = x_T.shape
        device = x_T.device
        
        x_t = x_T # Start at the maximally degraded state
        
        # We need the original k-space for Data Consistency
        if enforce_dc:
            orig_k_space = torch.fft.fft(self._to_complex(x_T), dim=-1)

        # Iterate backward from num_timesteps down to 1
        for time_step in reversed(range(1, self.num_timesteps + 1)):
            
            # Create a batched tensor of the current timestep
            t = torch.full((b,), time_step, device=device, dtype=torch.long)
            t_minus_1 = torch.full((b,), time_step - 1, device=device, dtype=torch.long)
            
            # 1. Predict the ground truth x_0 using the UNet
            x_0_pred = self.denoise_fn(x_t, t)
            
            # --- Optional Data Consistency Step ---
            # If we know exactly which low frequencies were captured by the sensor,
            # we force the network's prediction to perfectly match those frequencies.
            if enforce_dc:
                pred_k = torch.fft.fft(self._to_complex(x_0_pred), dim=-1)
                
                # Simple DC: just average the low frequencies of the prediction 
                # with the actual hard measurement to prevent drift.
                # (In a full implementation, you would apply a binary mask here)
                pred_k = (pred_k + orig_k_space) / 2.0 
                x_0_pred = self._to_real(torch.fft.ifft(pred_k, dim=-1))
     

            # 2. If we are at the final step, we are done
            if time_step == 1:
                x_t = x_0_pred
                break
                
            # 3. The Cold Diffusion Update Rule
            # x_{t-1} = x_t - D(x_0, t) + D(x_0, t-1)
            deg_t = self.q_sample(x_0_pred, t)
            deg_t_minus_1 = self.q_sample(x_0_pred, t_minus_1)
            
            x_t = x_t - deg_t + deg_t_minus_1
            
        self.denoise_fn.train()
        return x_t