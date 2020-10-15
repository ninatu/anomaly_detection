import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
import argparse


N_TUMOR_SLIDES_FOR_VALIDATION = 4
PATCH_NAME_PAT = re.compile('(?P<image_name>.*)_(?P<crop_type>.*)_x_(?P<x>\d+)_y_(?P<y>\d+)_w_(?P<w>\d+)_h_(?P<h>\d+)')


def _filter_filenames(slides, filenames):
    filtered = []
    for filename in filenames:
        if PATCH_NAME_PAT.match(filename).group('image_name') in slides:
            filtered.append(filename)
    return filtered


def _save_split(filenames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f_out:
        f_out.writelines([filename + '\n' for filename in filenames])


def create_folds(normal_train_split, tumor_train_split, output_root, n_folds):
    folds_dir = os.path.join(output_root, 'folds', 'camelyon16')
    validation_classes_root = os.path.join(output_root, 'validation_classes')
    validation_classes_path = os.path.join(validation_classes_root, 'camelyon16.csv')

    os.makedirs(validation_classes_root, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)

    "==========================  CHOOSE SLIDES FOR VALIDATION ========================="

    if not os.path.exists(validation_classes_path):
        with open(tumor_train_split) as f_in:
            anomaly_filenames = [filename.strip() for filename in f_in.readlines()]

        tumor_slidenames = [PATCH_NAME_PAT.match(filename).group('image_name') for filename in anomaly_filenames]
        tumor_slidenames = np.unique(tumor_slidenames)
        np.random.shuffle(tumor_slidenames)
        validation_slidenames = tumor_slidenames[:N_TUMOR_SLIDES_FOR_VALIDATION]

        df = pd.DataFrame(np.array(validation_slidenames)[:, np.newaxis], columns=['Valid Slides'])
        df.to_csv(validation_classes_path, index=False)

    "===================== CREATE K-FOLD CROSS-VALIDATION SPLIT ========================"

    valid_slides_df = pd.read_csv(validation_classes_path)
    valid_anomaly_slides = valid_slides_df['Valid Slides'].values

    with open(tumor_train_split) as f_in:
        anomaly_filenames = [filename.strip() for filename in f_in.readlines()]

    with open(normal_train_split) as f_in:
        normal_filenames = [filename.strip() for filename in f_in.readlines()]
    normal_slides = [PATCH_NAME_PAT.match(filename).group('image_name') for filename in normal_filenames]
    normal_slides = np.array(normal_slides)

    normal_train_split_indexes, normal_test_split_indexes = list(zip(*KFold(n_splits=n_folds).split(normal_slides)))
    _, anomaly_test_split_indexes = list(zip(*KFold(n_splits=n_folds).split(valid_anomaly_slides)))

    for i_fold, (normal_train_indexes, normal_test_indexes, anomaly_test_indexes) in \
            enumerate(zip(normal_train_split_indexes, normal_test_split_indexes, anomaly_test_split_indexes)):

        fold_dir = os.path.join(folds_dir, 'healthy', str(i_fold))
        os.makedirs(fold_dir, exist_ok=True)

        normal_train_filenames = _filter_filenames(normal_slides[normal_train_indexes], normal_filenames)
        normal_test_filenames = _filter_filenames(normal_slides[normal_test_indexes], normal_filenames)
        anomaly_test_filenames = _filter_filenames(valid_anomaly_slides[anomaly_test_indexes], anomaly_filenames)

        np.random.shuffle(normal_train_filenames)
        val_n = int(0.2 * len(normal_train_filenames))
        normal_val_filenames = normal_train_filenames[:val_n]
        normal_train_filenames = normal_train_filenames[val_n:]

        _save_split(normal_train_filenames, os.path.join(fold_dir, 'normal', 'train'))
        _save_split(normal_val_filenames, os.path.join(fold_dir, 'normal', 'val'))
        _save_split(normal_test_filenames, os.path.join(fold_dir, 'normal', 'test'))
        _save_split(anomaly_test_filenames, os.path.join(fold_dir, 'anomaly', 'test'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--normal_train_split",
                        type=str,
                        default='./folds/train_test_split/camelyon16/healthy/normal/train',
                        help='normal_train_split')
    parser.add_argument("--tumor_train_split",
                        type=str,
                        default='./folds/train_test_split/camelyon16/healthy/anomaly/train',
                        help='tumor_train_split')
    parser.add_argument("-o", "--output_root", required=True, type=str, help='output_root')
    parser.add_argument("-n", "--n_folds", type=int, default=3, help='n_folds')

    args = parser.parse_args()

    create_folds(args.normal_train_split, args.tumor_train_split, args.output_root, args.n_folds)
