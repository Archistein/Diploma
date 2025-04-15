import torch
import torch.nn as nn
from math import ceil, prod


class ConvBlock(nn.Module):
    """
    Convolutional block for VAE
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int],
                 stride: int | tuple[int],
                 padding: int | tuple[int]
                ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        num_groups = min(32, max(1, out_channels // 4)) 
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.elu = nn.ELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.gn(out)
        out = self.elu(out)

        return out
    

class ConvTpBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int],
                 stride: int | tuple[int],
                 padding: int | tuple[int],
                 output_padding: int | tuple[int],
                 first: bool = False
                ) -> None:
        super().__init__()

        if first:
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
        out = self.convt(x)
        out = self.gn(out)
        out = self.elu(out)

        return out
    

class Encoder(nn.Module):
    def __init__(self, 
                 img_size: tuple[int],
                 in_channels: int,
                 latent_dim: int,
                 hidden_layers: tuple[int]
                ) -> None:
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        for block in self.encoder:
            out = block['conv1'](x)
            x = block['conv2'](out + x)
        
        out = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 origin_shape: tuple[int],
                 hidden_layers: tuple[int],
                 out_pad: tuple[int] = (1, 1)
                ) -> None:
        super().__init__()

        self.fc = nn.Linear(latent_dim, prod(origin_shape))

        self.reshape = lambda z: z.view(-1, *origin_shape)

        self.decoder = nn.ModuleList()

        for i, (curr, next) in enumerate(zip(hidden_layers, hidden_layers[1:] + (hidden_layers[-1],))):
            self.decoder.append(nn.ModuleDict({
                'conv': ConvBlock(curr, curr, kernel_size=3, stride=1, padding=1),
                'convt': ConvTpBlock(curr, next, kernel_size=3, stride=(2 if not i else 1), padding=1, output_padding=(out_pad if not i else 1), first=(not i))
            }))
        
        self.head = nn.Conv2d(hidden_layers[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = self.reshape(z)

        for block in self.decoder:
            out = block['conv'](z)
            z = block['convt'](out + z)

        return self.head(z)
    

class VAE(nn.Module):
    def __init__(self, 
                 img_size: tuple[int],
                 in_channels: int,
                 latent_dim: int = 512,
                 hidden_layers: tuple[int] = (32, 64, 128, 256, 512)
                ) -> None:
        super().__init__()
        
        out_pad = (int(img_size[0] % len(hidden_layers) == 0), int(img_size[1] % len(hidden_layers) == 0))

        self.latent_dim = latent_dim
        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_layers)
        self.decoder = Decoder(in_channels, latent_dim, self.encoder.origin_shape, hidden_layers[::-1], out_pad)
    
    def forward(self, x: torch.Tensor, stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if stochastic else mu
        
        return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def sample(self, num: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num, self.latent_dim, device=device)

        return self.decoder(z)