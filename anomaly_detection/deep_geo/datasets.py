"""
Copyright (c) 2018 izikgo
Copyright (c) 2020 ninatu

This file is a modified file of the project: https://github.com/izikgo/AnomalyDetectionTransformations,
which was released under MIT License.
Go to https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/LICENSE for full license details.
"""

import numpy as np
import tqdm
import torch
from anomaly_detection.utils.datasets import DATASETS, DatasetType
from anomaly_detection.utils.transforms import TRANSFORMS
from torch.utils.data import DataLoader
import keras
from keras.backend import cast_to_floatx

torch.multiprocessing.set_sharing_strategy('file_system')


class GEOTransformedDataset:
    def __init__(self, dataset, geo_transformer):
        self._dataset = dataset
        self._geo_transformer = geo_transformer

    def __getitem__(self, index):
        i_image = index // self._geo_transformer.n_transforms
        i_transform = index % self._geo_transformer.n_transforms

        image = self._dataset[i_image].numpy().transpose(1, 2, 0)
        image = self._geo_transformer._transformation_list[i_transform](image)
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy())
        return image, i_transform

    def __len__(self):
        return len(self._dataset) * self._geo_transformer.n_transforms


def load_dataset(geo_transformer, dataset_type, dataset_kwargs, transform_kwargs):
    transform = TRANSFORMS[DatasetType(dataset_type)](**transform_kwargs)
    dataset = DATASETS[DatasetType(dataset_type)](**dataset_kwargs, transform=transform)
    geo_trans_dataset = GEOTransformedDataset(dataset, geo_transformer)

    shape = geo_trans_dataset[0][0].shape
    dtype = geo_trans_dataset[0][0].numpy().dtype

    data = np.empty((len(geo_trans_dataset), shape[0], shape[1], shape[2]), dtype=dtype)
    labels = np.empty(len(geo_trans_dataset))
    dataloader = DataLoader(geo_trans_dataset, batch_size=32, shuffle=False, drop_last=False,
                            num_workers=0, pin_memory=False)

    shift = 0
    for x, y in tqdm.tqdm(dataloader):
        batch_size = x.shape[0]
        data[shift:shift + batch_size] = x.numpy()
        labels[shift:shift + batch_size] = y.numpy()
        shift += batch_size
    return cast_to_floatx(data), labels
