import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torchvision import datasets
import argparse


def _get_target_indexes(dataset, target_label):
    target_indexes = []
    for index, (_, label) in enumerate(dataset):
        if label == target_label:
            target_indexes.append(index)
    return np.array(target_indexes)


def create_folds(svhn_root, output_root, n_folds):

    folds_dir = os.path.join(output_root, 'folds', 'svhn')
    validation_classes_root = os.path.join(output_root, 'validation_classes')
    validation_classes_path = os.path.join(validation_classes_root, 'svhn.csv')

    os.makedirs(validation_classes_root, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)

    dataset = datasets.SVHN(root=svhn_root, split='train', download=True)
    classes = np.unique(dataset.labels)
    n_classes = len(classes)

    "====================== GENERATE CLASSES FOR VALIDATION ======================"

    if not os.path.exists(validation_classes_path):
        df = pd.DataFrame(columns=['class', 'class_name', 'valid_class', 'valid_class_name'])

        for _class in range(n_classes):
            available_classes = list(range(10))
            available_classes.remove(_class)

            valid_class = np.random.choice(available_classes)
            df.loc[_class] = [_class, classes[_class], valid_class, classes[valid_class]]

        df.to_csv(validation_classes_path, index=False)

    "====================== CREATE K-FOLD CROSS-VALIDATION SPLIT ======================"

    valid_classes_df = pd.read_csv(validation_classes_path)
    valid_classes_df.set_index('class')

    for _class in range(n_classes):
        anomaly_class = valid_classes_df.loc[_class]['valid_class']

        normal_indexes = _get_target_indexes(dataset, _class)
        valid_anomaly_indexes = _get_target_indexes(dataset, anomaly_class)

        normal_train_subindexes, normal_test_subindexes = list(zip(*KFold(n_splits=n_folds).split(normal_indexes)))
        _, anomaly_test_subindexes = list(zip(*KFold(n_splits=n_folds).split(valid_anomaly_indexes)))

        for i_fold, (normal_train_subindex, normal_test_subindex, anomaly_test_subindex) in \
                enumerate(zip(normal_train_subindexes, normal_test_subindexes, anomaly_test_subindexes)):
            fold_dir = os.path.join(folds_dir, str(_class), str(i_fold))
            os.makedirs(fold_dir, exist_ok=True)

            np.save(os.path.join(fold_dir, 'test_normal'), normal_indexes[normal_test_subindex])
            np.save(os.path.join(fold_dir, 'test_anomaly'), valid_anomaly_indexes[anomaly_test_subindex])

            train_normal_indexes = normal_indexes[normal_train_subindex]
            np.random.shuffle(train_normal_indexes)

            val_n = int(0.2 * len(train_normal_indexes))
            val_normal_indexes = train_normal_indexes[:val_n]
            train_normal_indexes = train_normal_indexes[val_n:]

            np.save(os.path.join(fold_dir, 'train'), train_normal_indexes)
            np.save(os.path.join(fold_dir, 'val'), val_normal_indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--svhn_root",
                        type=str,
                        default='./data/data/svhn',
                        help='svhn_root')
    parser.add_argument("-o", "--output_root",
                        required=True,
                        type=str)
    parser.add_argument("-n", "--n_folds",
                        type=int,
                        default=3,
                        help='n_folds')

    args = parser.parse_args()

    svhn_root = args.svhn_root
    output_root = args.output_root
    n_folds = args.n_folds

    create_folds(svhn_root, output_root, n_folds)
