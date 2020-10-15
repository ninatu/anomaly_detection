import argparse
import yaml
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from anomaly_detection.piad.train import Trainer
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS


def evaluate(config):
    batch_size = config['test_batch_size']
    results_root = config['results_root']
    model_path = config['test_model_path']

    os.makedirs(results_root, exist_ok=True)

    print(yaml.dump(config, default_flow_style=False))

    print("Starting model evaluation ...")

    enc, gen, image_rec_loss, niter = \
        Trainer.load_anomaly_detection_model(torch.load(model_path))
    enc, gen, image_rec_loss = enc.cuda().eval(), gen.cuda().eval(), image_rec_loss.cuda().eval()

    dataset_type = config['test_datasets']['normal']['dataset_type']
    dataset_kwargs = config['test_datasets']['normal']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['normal']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    normal_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    dataset_type = config['test_datasets']['anomaly']['dataset_type']
    dataset_kwargs = config['test_datasets']['anomaly']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['anomaly']['transform_kwargs']
    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    anomaly_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, normal_dataset, batch_size)
    an_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, anomaly_dataset, batch_size)

    y_true = np.concatenate((np.zeros_like(norm_anomaly_scores), np.ones_like(an_anomaly_scores)))
    y_pred = np.concatenate((np.array(norm_anomaly_scores), np.array(an_anomaly_scores)))

    roc_auc = roc_auc_score(y_true, y_pred)

    output_path = os.path.join(results_root, 'results.csv')
    results = pd.DataFrame([[niter, roc_auc]], columns=['niter', 'ROC AUC'])

    print("Model evaluation is complete. Results: ")
    print(results)
    results.to_csv(output_path, index=False)


def predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    data_loader = tqdm(data_loader)

    image_rec_loss.set_reduction('none')

    anomaly_scores = []
    for images in data_loader:
        images = images.cuda()
        with torch.no_grad():
            rec_images = gen(enc(images)).detach()
            cur_anomaly_scores = image_rec_loss(images, rec_images)
        anomaly_scores.extend(cur_anomaly_scores.detach().cpu().numpy())

    return anomaly_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str, nargs='*', help='Config paths')

    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        evaluate(config)


if __name__ == '__main__':
    main()
