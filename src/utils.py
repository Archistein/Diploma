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
    """"Iterable dataset for loading and transforming face images on-the-fly"""

    def __init__(self, 
                 iterable: iterable_dataset.IterableDataset,
                 transforms: A.Compose
                ) -> None:
        """Initializes the FacesIterDataset

        Args:
            iterable (iterable_dataset.IterableDataset): The streaming dataset to iterate over
            transforms (A.Compose): Albumentations transformations to apply to each image
        """

        self.iterable = iterable
        self.transforms = transforms

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Returns an iterator over the transformed dataset.

        Yields:
            (torch.Tensor): Transformed image tensor.
        """

        for item in self.iterable:
            image = np.array(item['image'])
            image = self.transforms(image=image)['image']

            yield image


class FacesDataset(torch.utils.data.Dataset):
    """Standard dataset for face images"""

    def __init__(self,
                 dataset: Dataset,
                 transforms: A.Compose
                ) -> None:
        """Initializes the FacesDataset

        Args:
            dataset (Dataset): HuggingFace dataset object
            transforms (A.Compose): Albumentations transformations to apply to each image
        """

        self.dataset = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the number of samples in the dataset

        Returns:
            (int): Total number of samples
        """

        return len(self.dataset)
  
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves and transforms a single sample from the dataset

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            (torch.Tensor): Transformed image tensor
        """

        image = np.array(self.dataset[idx]['image'])
        image = self.transforms(image=image)['image']

        return image


class AnnotatedFacesDataset(torch.utils.data.Dataset):
    """Dataset that returns both face images and their attribute annotations"""

    def __init__(self,
                 dataset: Dataset,
                 transforms: A.Compose
                ) -> None:
        """Initializes the AnnotatedFacesDataset

        Args:
            dataset (Dataset): HuggingFace dataset with attribute annotations
            transforms (A.Compose): Albumentations transformations to apply to each image
        """

        self.dataset = dataset
        self.transforms = transforms
        self.features = list(dataset.features.keys())[2:]

    def __len__(self) -> int:
        """Returns the number of samples in the dataset

        Returns:
            (int): Total number of samples
        """

        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, bool]]:
        """Retrieves a sample and its corresponding annotations

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            (tuple[torch.Tensor, dict[str, bool]]): A tuple containing the transformed image tensor and a dictionary of attribute annotations
        """

        image = np.array(self.dataset[idx]['image'])
        image = self.transforms(image=image)['image']

        return image, {key: self.dataset[idx][key] for key in self.features}


def denormalize(img: torch.Tensor, mean: float = config['calc_mean'], std: float = config['calc_std']) -> np.ndarray:
    """Reverses normalization on a tensor image

    Args:
        img (torch.Tensor): Normalized image tensor
        mean (float): Mean used for normalization
        std (float): Standard deviation used for normalization

    Returns:
        (np.ndarray): Denormalized image as a NumPy array in [0, 255] range
    """

    return np.clip((img.cpu().detach().numpy() * std + mean)*255, 0, 255).astype(np.uint8)


def get_dataset(test_size: float = 0.05, seed: int = config['seed']) -> Dataset:
    """Loads and splits the CelebA dataset

    Args:
        test_size (float): Fraction of dataset to reserve for testing
        seed (int): Random seed for reproducibility

    Returns:
        (Dataset): A dictionary with 'train' and 'test' dataset splits
    """

    dataset = load_dataset("nielsr/CelebA-faces", split='train').train_test_split(test_size, seed=seed)
    #annotated_ds = load_dataset("flwrlabs/celeba", streaming=True) # For searching semantic vectors

    return dataset


def get_transforms(
        img_height: int = config['img_height'], 
        img_width: int = config['img_width'], 
        calc_mean: float = config['calc_mean'], 
        calc_std: float = config['calc_std'],
    ) -> tuple[A.Compose, A.Compose]:
    """Creates data augmentation pipelines for training and validation

    Args:
        img_height (int): Target image height after cropping
        img_width (int): Target image width after cropping
        calc_mean (float): Mean value for normalization
        calc_std (float): Standard deviation for normalization

    Returns:
        (tuple[A.Compose, A.Compose]): A tuple of Albumentations Compose objects for training and validation
    """

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
    """Creates PyTorch dataloaders for training and validation datasets

    Args:
        dataset (Dataset): Dictionary containing 'train' and 'test' datasets
        train_transforms (A.Compose): Transformations for training data
        val_transforms (A.Compose): Transformations for validation data
        batch_size (int): Number of samples per batch

    Returns:
        (tuple[DataLoader, DataLoader]): Training and validation DataLoaders
    """

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