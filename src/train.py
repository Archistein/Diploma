import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_msssim import SSIM
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from model import VAE
import utils


vgg_losses = []
ssim_losses = []
logcosh_losses = []

ssim_module = SSIM(data_range=1.0, size_average=True, channel=1).to(utils.device)
VGG_features = vgg16(weights=VGG16_Weights.DEFAULT).eval().features
VGG_features.to(utils.device)

for param in VGG_features.parameters():
    param.requires_grad = False


def extract_features(x: torch.Tensor) -> list[torch.Tensor]:
    """Extracts intermediate features from VGG16 for perceptual loss

    Args:
        x (torch.Tensor): Input image tensor with shape (B, 3, H, W)

    Returns:
        (list[torch.Tensor]): List of feature maps from selected VGG16 layers
    """

    features_idx = [3, 6, 10, 14]
    features = []
    out = x

    for (k, module) in VGG_features._modules.items():
        if int(k) > features_idx[-1]: break
        out = module(out)
        if int(k) in features_idx:
            features.append(out)

    return features


def ssim_denorm(tensor: torch.Tensor, mean=utils.config['calc_mean'], std=utils.config['calc_std']) -> torch.Tensor:
    """Applies inverse normalization to a tensor for SSIM calculation

    Args:
        tensor (torch.Tensor): Normalized input tensor
        mean (float): Mean used in original normalization
        std (float): Std used in original normalization

    Returns:
        (torch.Tensor): Denormalized tensor
    """

    return tensor * std + mean


def loss(
    reconstr: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor, 
    logvar: torch.Tensor,
    alpha: float = 1,
    beta: float = 1,
    lmbd: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the combined KLd, reconstruction, SSIM, and VGG perceptual loss

    Args:
        reconstr (torch.Tensor): Reconstructed image from VAE
        target (torch.Tensor): Ground truth image
        mu (torch.Tensor): Latent mean from VAE encoder
        logvar (torch.Tensor): Latent log-variance from VAE encoder
        alpha (float): Weight for reconstruction loss
        beta (float): Weight for SSIM loss
        lmbd (float): Weight for VGG perceptual loss

    Returns:
        (tuple[torch.Tensor, torch.Tensor]):
            - Total weighted loss;
            - Mean KL divergence regularization term.
    """

    vgg_loss = torch.tensor(0.0, device=utils.device, requires_grad=True)
    
    input_features = extract_features(target.repeat(1, 3, 1, 1))
    reconstruction_features = extract_features(reconstr.repeat(1, 3, 1, 1))

    for i, (inp, rec) in enumerate(zip(input_features, reconstruction_features)):
        inp = inp / (inp.std() + 1e-8)
        rec = rec / (rec.std() + 1e-8)
        vgg_loss = vgg_loss + (0.95 ** i) * F.mse_loss(inp, rec)

    reconstruction_loss = torch.mean(torch.log(torch.cosh(reconstr - target)))
    regularization_term = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)

    generated = ssim_denorm(reconstr)
    target = ssim_denorm(target)
    
    generated = torch.clamp(generated, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    ssim_loss = 1 - ssim_module(generated, target)

    vgg_losses.append(vgg_loss.item())
    ssim_losses.append(ssim_loss.item())
    logcosh_losses.append(reconstruction_loss.item())
    
    return alpha * reconstruction_loss + beta * ssim_loss + lmbd * vgg_loss, regularization_term.mean()


@torch.inference_mode
def evaluate(
    vae: VAE, 
    beta: float,
    dataloader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    """Evaluates the VAE model on a validation set

    Args:
        vae (VAE): The trained variational autoencoder model
        beta (float): Weight for the KL divergence term
        dataloader (DataLoader): DataLoader for validation data

    Returns:
        (tuple[float, float]): Average reconstruction and KL divergence losses
    """

    vae.eval()

    running_rec_loss = 0
    running_kl_loss = 0
    total_samples = 0

    for inputs in (pbar := tqdm(dataloader, desc=f'Validation')):
    
        inputs = inputs.to(utils.config.device)

        reconstructions, mu, logvar = vae(inputs, stochastic=False)

        reconstruction_loss, kl_loss = loss(reconstructions, inputs, mu, logvar, False)

        running_rec_loss += reconstruction_loss.item() * inputs.size(0)
        running_kl_loss += kl_loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        avg_rec_loss = running_rec_loss / total_samples
        avg_kl_loss = running_kl_loss / total_samples
        avg_elbo_loss = avg_rec_loss + beta * avg_kl_loss

        pbar.set_description(f'Validation | Rec loss: {avg_rec_loss:.05f} | KL loss: {avg_kl_loss:.05f} | Elbo loss: {avg_elbo_loss:.05f}')

    vae.train()

    return avg_rec_loss, avg_kl_loss


def fit(
    vae: VAE,
    config: OmegaConf,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
) -> None:
    """Trains the VAE model using a custom loss function

    Args:
        vae (VAE): The VAE model to be trained
        config (OmegaConf): Training hyperparameters and settings
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
    """

    vae.train()
    vae.to(utils.device)

    delta = config['delta_beta']
    max_beta = config['max_beta']
    min_beta = config['min_beta']
    beta = min_beta

    optimizer = optim.AdamW(vae.parameters(), config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['scheduler_gamma'])

    rec_losses = []
    kl_losses = []
    elbo_losses = []

    min_elbo = float('inf')

    for e in range(config['epoch']):
        running_rec_loss = 0
        running_kl_loss = 0
        total_samples = 0

        for i, inputs in (pbar := tqdm(enumerate(train_dataloader), desc=f'Epoch: {e+1}')):
        
            inputs = inputs.to(utils.device)

            reconstructions, mu, logvar = vae(inputs)

            reconstruction_loss, kl_loss = loss(reconstructions, inputs, mu, logvar)
            elbo_loss = reconstruction_loss + beta * kl_loss

            running_rec_loss += reconstruction_loss.item() * inputs.size(0)
            running_kl_loss += kl_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            optimizer.zero_grad()
            elbo_loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), config['grad_clip'])
            optimizer.step()

            avg_rec_loss = running_rec_loss / total_samples
            avg_kl_loss = running_kl_loss / total_samples
            avg_elbo_loss = avg_rec_loss + avg_kl_loss * 1e-4

            pbar.set_description(f'Epoch: {e+1} | Rec loss: {avg_rec_loss:.05f} | KL loss: {avg_kl_loss:.05f} | Beta = {beta}')

            if (i % 300 == 0 and i != 0):
                beta += delta * np.cos(e)
                
                if beta < min_beta: 
                    beta = min_beta
                if beta > max_beta:
                    beta = max_beta
                
                if avg_elbo_loss < min_elbo:
                    min_elbo = avg_elbo_loss
                    torch.save(vae.state_dict(), 'params.pt')

                running_rec_loss = 0
                running_kl_loss = 0
                total_samples = 0

            rec_losses.append(avg_rec_loss)
            kl_losses.append(avg_kl_loss)
            elbo_losses.append(avg_elbo_loss)
        
        val_rec_loss, val_kl_loss = evaluate(vae, beta, val_dataloader)

        torch.save(vae.state_dict(), 'last_params.pt')

        scheduler.step()