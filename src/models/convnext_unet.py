import math 
import torch
import torch.nn as nn
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

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


class LinearAttention(nn.Module):
    """
    Linear attention mechanism that reduces the computational complexity from O(N^2) to O(N)
    by using a kernel function to approximate the softmax operation in traditional attention.

    Implementation based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9423033 
    repo: https://github.com/cmsflash/efficient-attention.

    Args:
        dim: The number of channels in the input tensor.
        heads: The number of attention heads.
        dim_head: The dimensionality of each attention head.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Project to q, k, v using a 1x1 convolutional layer
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Reshape from (Batch, Channels, height, width) to (Batch, Heads, Head_Dim, SequenceLenght)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # Scale the query queries (stabilization)
        q = q * self.scale

        # Turn the keys into probabilities using a softmax function, this is done across the sequence length dimension,
        # which is the last dimension in the reshaped tensors. This allows the model to focus on different 
        # parts of the input when computing the attention.
        k = k.softmax(dim=-1)

        # Compute the "Context" matrix: K^T * V
        # 'n' is sequence length (H*W), 'd' is Key feature dim, 'e' is Value feature dim.
        # We sum out 'n', leaving a tiny (Head_Dim x Head_Dim) matrix for each head.
        # Sorry for the einsum notation, it is not readable but it is very efficient,
        # it is basically doing a batch matrix multiplication between the transposed keys
        # and the values, resulting in a context matrix that captures the relationships 
        # between the keys and values across the sequence length. 
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Multiply the (Head_Dim x Head_Dim) context matrix by the Queries.
        # This is q * context, where q has shape (Batch, Heads, Head_Dim, SequenceLength) and context has shape (Batch, Heads, Head_Dim, Head_Dim).
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # Reshape back into an image format (Batch, Channels, Height, Width)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        # Final 1x1 convolution projection
        return self.to_out(out)


class Unet(nn.Module):
    """
    U-Net architecture for image-to-image translation tasks. This is a standard ConvNexT U-Net
    with linear attention layers and time embeddings added to the ConvNext blocks. 

    Args:
        dim: The number of channels in the input tensor.
        out_dim: The number of channels in the output tensor = channels (almost always)
        dim_mults: A tuple that defines the scaling of the number of channels in each layer of the U-Net.
        channels: The number of channels in the input image (2 for complex field, real and imaginary).
        with_time_emb: Whether to use time embeddings in the ConvNext blocks.
        residual: Whether to add a residual connection from the input to the output of the U-Net.
    """
    def __init__(
            self,
            dim: int,
            out_dim: int = None,
            dim_mults: tuple = (1, 2, 4, 8),
            channels: int = 2, # Complex field has 2 channels, real and imaginary
            with_time_emb: bool = True,
            residual: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual

        print('Using time embeddings:', with_time_emb)

        # Add an initial convolution to map input channels to the base feature dimension
        self.init_conv = nn.Conv2d(channels, dim, kernel_size=7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim

            # Time embedding MLP, this is a small feedforward network that takes the sinusoidal
            #  positional embeddings of the time steps and transforms them into a format that can
            #  be used by the ConvNext blocks. The output of this MLP is added to the feature maps
            #  in the ConvNext blocks, allowing the model to learn how to handle different time 
            #  steps effectively. Time steps symbolize the level of deterioration due to the diffusion function
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # Downsampling Layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim),
                    ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity()
                ])
            )

        # Middle Layers
        mid_dim = dims[-1] # Mid Dim is just last dimension after downsampling
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling Layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity()
                ])
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, kernel_size=1)
        )


    # This just applies the forward pass through the U-Net architecture, 
    # it is a standard U-Net with skip connections and time embeddings added
    #  to the ConvNext blocks. The input is passed through the downsampling layers, 
    # then through the middle layers, and finally through the upsampling layers, with 
    # skip connections between the corresponding downsampling and upsampling layers.
    #  The output of the final convolutional layer is returned as the output of the U-Net.
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        orig_x = x
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        if self.residual:
            return orig_x + self.final_conv(x)
        
        return self.final_conv(x)



        