import math 
import torch
import torch.nn as nn
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None

class EMA():
    """
    Maintains a copy of the model with Exponential Moving Average (EMA) weights. 
    This is often used in training to stabilize the model and improve performance during inference.

    Attributes:
        beta (float): The decay rate for the EMA. 
    """
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module):
        """
        Updates the EMA model's weights by blending them with the current model's weights.

        Args:
            ema_model: The model that maintains the EMA weights.
            model: The current model whose weights are being updated.
        """
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight = ema_params.data
            new_weight = current_params.data

            ema_params.data = self.update_average(old_weight, new_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """
        Computes the new EMA weight by blending the old EMA weight with the new weight.

        Args:
            old: The current EMA weight.
            new: The new weight from the current model.

        Returns:
            The updated EMA weight.
        """
        if old is None:
            return new
        
        return old * self.beta + (1 - self.beta) * new
    

class Residual(nn.Module):
    """
    Residual block that adds the input to the output of a given function.

    Attributes:
        fn: Function that computes the given input
    """
    def __init__(self, fn: nn.Module | callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Forward pass does f(x) + x, this is the residual step
        """
        return self.fn(x, *args, **kwargs) + x
    


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding module that generates positional embeddings based on sine and cosine functions.

    These are the same used in the original transformer paper to encode positional information. We will use them 
    to encode the time step in the diffusion process, allowing the model to learn how to handle different time steps effectively.

    Attributes:
        dim: The dimensionality of the positional embeddings.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Calculates the sinusoidal positional embeddings and loads it to the same device as the input tensor x.

        Args:
            x: The input tensor, typically representing time steps in the diffusion process.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim: int):
    """
    Learnable transpose convolution layer to upsample the input feature maps by a factor of 2. 

    The spatial dimension of a 2D transpose convolution in pytorch is calculated as follows:
    H_out = (H_in - 1) * S - 2P + K, with P being the padding, K being the kernel size, and S being the stride.
    """
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


def Downsample(dim: int):
    """
    Learnable downsampling convolutional layer that reduces the spatial dimensions of the input feature maps by a factor of 2.

    The spatial dimension of a 2D convolution in pytorch is calculated as follows:
    H_out = [H_in + 2P - K / S] + 1, with P being the padding, K being the kernel size, and S being the stride.
    """
    return nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)


class ComplexSafeLayerNorm(nn.Module):
    """
    Layer norm that calculates the statistics across Channels, Height, and Width, but not across the Re Im. 

    Real and Imaginary are scaled by the exact scalar

    Args:
        dim: The number of channels in the input tensor (2 for complex field).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate mean and variance across Channels, Height, and Width
        """
        var = torch.var(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)

        return (x - mean) / torch.sqrt(var + self.eps) * self.gain + self.bias
    
class PreNorm(nn.Module):
    """
    Pre-normalization layer that applies normalization before the given function.

    Attributes:
        dim: The number of channels in the input tensor.
        fn: The function to be applied after normalization.
    """
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = ComplexSafeLayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return self.fn(self.norm(x))


class ConvNextBlock(nn.Module):
    """
    Taken exactly from the original ConvNext paper, but with the addition of a time embedding.
    https://arxiv.org/abs/2201.03545 
    """
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int = None, mult: int = 2, norm: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Bottleneck convolutional layer, akin to MLP in the original ConvNext block, but with a time embedding added if provided
        self.net = nn.Sequential(
            ComplexSafeLayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.ds_conv(x)
        
        if exists(self.mlp):
            assert exists(time_emb)

            condition = self.mlp(time_emb)

            # Rearrange time embedding to be added to the feature maps, this is done by adding two extra dimensions to the time embedding and then broadcasting it across the spatial dimensions of the feature maps.
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)

        return h + self.res_conv(x)