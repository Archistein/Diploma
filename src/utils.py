from typing import Iterator
import torch
import numpy as np
from datasets import load_dataset, iterable_dataset, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os


config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
config = OmegaConf.load(config_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FacesIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 iterable: iterable_dataset.IterableDataset,
                 transforms: A.Compose
                ) -> None:
        self.iterable = iterable
        self.transforms = transforms

    def __iter__(self) -> Iterator[torch.Tensor]:
        for item in self.iterable:
            image = np.array(item['image'])
            image = self.transforms(image=image)['image']

            yield image


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset: Dataset,
                 transforms: A.Compose
                ) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)
  
    def __getitem__(self, idx: int) -> torch.Tensor:
        image = np.array(self.dataset[idx]['image'])
        image = self.transforms(image=image)['image']

        return image


def denormalize(img: torch.Tensor, mean: float = config['calc_mean'], std: float = config['calc_std']) -> np.ndarray:
    return np.clip((img.cpu().detach().numpy() * std + mean)*255, 0, 255).astype(np.uint8)


def get_dataset(test_size: float = 0.05, seed: int = config['seed']) -> Dataset:
    dataset = load_dataset("nielsr/CelebA-faces", split='train').train_test_split(test_size, seed=seed)
    #annotated_ds = load_dataset("flwrlabs/celeba", streaming=True) # For searching semantic vectors

    return dataset


def get_transforms(
        img_height: int = config['img_height'], 
        img_width: int = config['img_width'], 
        calc_mean: float = config['calc_mean'], 
        calc_std: float = config['calc_std'],
    ) -> tuple[A.Compose, A.Compose]:

    transforms_list = [
        A.CenterCrop(img_height, img_width, p=1),
        A.ToGray(1, p=1),
        A.Normalize(mean=calc_mean, std=calc_std),
        ToTensorV2()
    ]

    train_transforms = A.Compose([A.HorizontalFlip(p=0.5)] + transforms_list)
    val_transforms = A.Compose(transforms_list)


    return train_transforms, val_transforms


def get_dataloaders(
        dataset: Dataset, 
        train_transforms: A.Compose, 
        val_transforms: A.Compose, 
        batch_size: int = config['batch_size']
    ) -> tuple[DataLoader, DataLoader]:

    train_dataloader = DataLoader(FacesDataset(dataset['train'], train_transforms),
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4)
    
    val_dataloader = DataLoader(FacesDataset(dataset['test'], val_transforms),
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=4)
    
    return train_dataloader, val_dataloader