import os
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import tqdm
from enum import Enum
import PIL.Image

__all__ = ['DatasetType', 'DATASETS', 'CIFAR10Dataset', 'NIHDataset', 'SVHNDataset']


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, target_classes=None, target_indexes_path=None):
        super().__init__()

        if split == 'train':
            self._dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        else:
            self._dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

        if (target_classes is not None) and (target_indexes_path is not None):
            raise ValueError("You must specify either 'target_classes' either 'target_indexes_path',"
                             "but not both")
        if target_classes is not None:
            self._target_indexes = []
            for index, label in enumerate(self._dataset.targets):
                if label in target_classes:
                    self._target_indexes.append(index)
        elif target_indexes_path is not None:
            self._target_indexes = np.load(target_indexes_path)
        else:
            self._target_indexes = list(range(len(self._dataset)))

    def __getitem__(self, index):
        image, _ = self._dataset[self._target_indexes[index]]
        return image

    def __len__(self):
        return len(self._target_indexes)


class SVHNDataset(Dataset):
    def __init__(self, root, split, transform=None, target_classes=None, target_indexes_path=None):
        super().__init__()

        self._dataset = datasets.SVHN(root=root, split=split, transform=transform, download=True)

        if (target_classes is not None) and (target_indexes_path is not None):
            raise ValueError("You must specify either 'target_classes' either 'target_indexes_path',"
                             "but not both")
        if target_classes is not None:
            self._target_indexes = []
            for index, (_, label) in enumerate(self._dataset):
                if label in target_classes:
                    self._target_indexes.append(index)
        elif target_indexes_path is not None:
            self._target_indexes = np.load(target_indexes_path)
        else:
            self._target_indexes = list(range(len(self._dataset)))

    def __getitem__(self, index):
        image, _ = self._dataset[self._target_indexes[index]]
        return image

    def __len__(self):
        return len(self._target_indexes)


class Camelyon16Dataset(Dataset):
    def __init__(self, image_root, split_root, split, transform=None, cache_data=False):
        super().__init__()

        self._image_root = image_root
        self._transform = transform

        split_info_path = os.path.join(split_root, split)
        with open(split_info_path) as f_in:
            self._image_filenames = [filename.strip() for filename in f_in.readlines()]

        self._cached_images = {}
        if cache_data:
            self._cache_data = False
            print('Loading dataset ... ')
            for index in tqdm.tqdm(range(len(self))):
                self._cached_images[index] = self[index]
        self._cache_data = cache_data

    def __getitem__(self, index):
        if self._cache_data:
            return self._cached_images[index]
        else:
            image_path = os.path.join(self._image_root, self._image_filenames[index])
            image = PIL.Image.open(image_path)
            if self._transform is not None:
                image = self._transform(image)
            return image

    def __len__(self):
        return len(self._image_filenames)


class NIHDataset(Dataset):
    def __init__(self, image_root, split_root, split, transform=None, cache_data=False):
        super().__init__()

        self._image_root = image_root
        self._transform = transform

        split_info_path = os.path.join(split_root, split)
        with open(split_info_path) as f_in:
            self._image_filenames = [filename.strip() for filename in f_in.readlines()]

        self._cached_images = {}
        if cache_data:
            self._cache_data = False
            print('Loading dataset ... ')
            for index in tqdm.tqdm(range(len(self))):
                self._cached_images[index] = self[index]
        self._cache_data = cache_data

    def __getitem__(self, index):
        if self._cache_data:
            return self._cached_images[index]
        else:
            image_path = os.path.join(self._image_root, self._image_filenames[index])
            image = PIL.Image.open(image_path)
            if self._transform is not None:
                image = self._transform(image)
            return image

    def __len__(self):
        return len(self._image_filenames)


class DatasetType(Enum):
    cifar10 = 'cifar10'
    camelyon16 = 'camelyon16'
    nih = 'nih'
    svhn = 'svhn'


DATASETS = {
    DatasetType.cifar10: CIFAR10Dataset,
    DatasetType.camelyon16: Camelyon16Dataset,
    DatasetType.nih: NIHDataset,
    DatasetType.svhn: SVHNDataset,
}
