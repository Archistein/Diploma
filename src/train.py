import torch
from torchvision.models import vgg16, VGG16_Weights
import utils

VGG_features = vgg16(weights=VGG16_Weights.DEFAULT).eval().features
VGG_features.to(utils.device)

for param in VGG_features.parameters():
    param.requires_grad = False

def extract_features(x: torch.Tensor) -> list[torch.Tensor]:
    features_idx = [3, 6, 10, 14]
    features = []
    out = x

    for (k, module) in VGG_features._modules.items():
        if int(k) > features_idx[-1]: break
        out = module(out)
        if int(k) in features_idx:
            features.append(out)
    return features