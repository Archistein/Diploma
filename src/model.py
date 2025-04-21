import torch
import torch.nn as nn
from math import ceil, prod


class ConvBlock(nn.Module):
    """Convolutional block with convolution, group normalization, and ELU activation"""

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int],
        padding: int | tuple[int]
    ) -> None:
        """Initializes the ConvBlock

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int | tuple[int]): Size of the convolutional kernel
            stride (int | tuple[int]): Stride of the convolution
            padding (int | tuple[int]): Padding added to both sides of the input
        """

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        num_groups = min(32, max(1, out_channels // 4)) 
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.elu = nn.ELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvBlock

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            (torch.Tensor): Output tensor after convolution, normalization, and activation.
        """

        out = self.conv(x)
        out = self.gn(out)
        out = self.elu(out)

        return out
    

class DeconvBlock(nn.Module):
    """Deconvolutional block with optional transpose convolution or upsampling + convolution"""

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int | tuple[int],
        padding: int | tuple[int],
        output_padding: int | tuple[int],
        transpose: bool = False
    ) -> None:
        """Initializes the DeconvBlocks

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (tuple[int]): Size of the convolutional kernel
            stride (int | tuple[int]): Stride of the convolution or upsampling
            padding (int | tuple[int]): Padding added to both sides of the input
            output_padding (int | tuple[int]): Additional size added to one side of the output
            transpose (bool): If True, use ConvTranspose2d; else use Upsample + Conv2d. Defaults to False.
        """

        super().__init__()

        if transpose:
            self.convt =  nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)
        else:
            self.convt = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            )     
        num_groups = min(32, max(1, out_channels // 4)) 
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.elu = nn.ELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DeconvBlock

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            (torch.Tensor): Output tensor after deconvolution, normalization, and activation
        """

        out = self.convt(x)
        out = self.gn(out)
        out = self.elu(out)

        return out
    

class Encoder(nn.Module):
    """Encoder for a Variational Autoencoder"""

    def __init__(
        self, 
        img_size: tuple[int],
        in_channels: int,
        latent_dim: int,
        hidden_layers: tuple[int]
    ) -> None:
        """Initializes the Encoder

        Args:
            img_size (tuple[int]): Size of input images (height, width)
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            hidden_layers (tuple[int]): Number of channels in each hidden layer
        """

        super().__init__()

        self.encoder = nn.ModuleList()
        
        for layer in hidden_layers:
            self.encoder.append(nn.ModuleDict({
                'conv1': ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                'conv2': ConvBlock(in_channels, layer, kernel_size=3, stride=2, padding=1)
            }))
            in_channels = layer
        
        self.origin_shape = hidden_layers[-1], ceil(img_size[0] / 2**len(hidden_layers)), ceil(img_size[1] / 2**len(hidden_layers))

        enc_dim = self.origin_shape[0] * self.origin_shape[1] * self.origin_shape[2]
        
        self.fc_mu = nn.Linear(enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Encoder

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Mean (mu) and log-variance (logvar) of the latent distribution
        """

        for block in self.encoder:
            out = block['conv1'](x)
            x = block['conv2'](out + x)
        
        out = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar
    

class Decoder(nn.Module):
    """Decoder for a Variational Autoencoder"""

    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        origin_shape: tuple[int],
        hidden_layers: tuple[int],
        out_pad: tuple[int] = (1, 1)
    ) -> None:
        """Initializes the Decoder

        Args:
            out_channels (int): Number of output channels (same as input image channels)
            latent_dim (int): Dimension of the latent space
            origin_shape (tuple[int]): Shape of the encoder's final feature map
            hidden_layers (tuple[int]): Number of channels in each hidden layer
            out_pad (tuple[int]): Output padding for the first deconvolution. Defaults to (1, 1)
        """

        super().__init__()

        self.fc = nn.Linear(latent_dim, prod(origin_shape))

        self.reshape = lambda z: z.view(-1, *origin_shape)

        self.decoder = nn.ModuleList()

        for i, (curr, next) in enumerate(zip(hidden_layers, hidden_layers[1:] + (hidden_layers[-1],))):
            self.decoder.append(nn.ModuleDict({
                'conv': ConvBlock(curr, curr, kernel_size=3, stride=1, padding=1),
                'convt': DeconvBlock(curr, 
                                     next, 
                                     kernel_size=3, 
                                     stride=(2 if not i else 1), 
                                     output_padding=(out_pad if not i else 1), 
                                     padding=1, 
                                     transpose=(not i)
                                    )
            }))
        
        self.head = nn.Conv2d(hidden_layers[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Decoder

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (batch_size, out_channels, height, width)
        """

        z = self.fc(z)
        z = self.reshape(z)

        for block in self.decoder:
            out = block['conv'](z)
            z = block['convt'](out + z)

        return self.head(z)
    

class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(
        self, 
        img_size: tuple[int],
        in_channels: int,
        latent_dim: int = 512,
        hidden_layers: tuple[int] = (32, 64, 128, 256, 512)
    ) -> None:
        """Initializes the VAE

        Args:
            img_size (tuple[int]): Size of input images (height, width)
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space. Defaults to 512
            hidden_layers (tuple[int]): Number of channels in each hidden layer. Defaults to (32, 64, 128, 256, 512)
        """

        super().__init__()
        
        out_pad = (int(img_size[0] % len(hidden_layers) == 0), int(img_size[1] % len(hidden_layers) == 0))

        self.latent_dim = latent_dim
        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_layers)
        self.decoder = Decoder(in_channels, latent_dim, self.encoder.origin_shape, hidden_layers[::-1], out_pad)
    
    def forward(self, x: torch.Tensor, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            stochastic (bool): If True, sample from latent distribution; else use mean. Defaults to True

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Reconstructed image, mean (mu), and log-variance (logvar).
        """

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if stochastic else mu
        
        return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution

        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log-variance of the latent distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def sample(self, num: int, device: torch.device) -> torch.Tensor:
        """Generate samples from random latent vectors

        Args:
            num (int): Number of samples to generate
            device (torch.device): Device to perform computation on

        Returns:
            torch.Tensor: Generated samples of shape (num, in_channels, height, width)
        """

        z = torch.randn(num, self.latent_dim, device=device)

        return self.decoder(z)