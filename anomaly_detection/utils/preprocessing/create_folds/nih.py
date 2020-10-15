import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import argparse


def _save_split(annotation, patients, labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    annotation = annotation[annotation['Patient ID'].isin(patients)]
    labels = set(labels)
    annotation = annotation[annotation.apply(lambda row: set(row['Finding Labels'].split('|')) == labels, axis=1)]

    with open(out_path, 'w') as f_out:
        f_out.writelines([filename + '\n' for filename in annotation['Image Index'].values])


def create_folds(path_to_data_entry, path_to_train_val_list, output_root, n_folds):
    folds_dir = os.path.join(output_root, 'folds', 'nih')
    validation_classes_root = os.path.join(output_root, 'validation_classes')
    validation_classes_path = os.path.join(validation_classes_root, 'nih.csv')

    os.makedirs(validation_classes_root, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)

    VIEWS = ["AP", "PA"]
    VALID_LABEL = 'Infiltration'

    "========================== GENERATE CLASSES FOR VALIDATION ========================="

    if not os.path.exists(validation_classes_path):
        df = pd.DataFrame([[VALID_LABEL]], columns=['Valid Labels'])
        df.to_csv(validation_classes_path, index=False)

    "===================== CREATE K-FOLD CROSS-VALIDATION SPLIT ========================"

    valid_labels_df = pd.read_csv(validation_classes_path)
    valid_anomaly_labels = valid_labels_df['Valid Labels'].values

    with open(path_to_train_val_list) as fin:
        image_names = list(map(lambda x: x.strip(), fin.readlines()))

    annotation = pd.read_csv(path_to_data_entry)
    annotation = annotation[annotation['Image Index'].isin(image_names)]

    for view in VIEWS:
        view_annotation = annotation[annotation['View Position'] == view]

        patients = list(view_annotation['Patient ID'].unique())
        patients = np.array(patients)
        np.random.shuffle(patients)

        for i_fold, (train_index, test_index) in enumerate(KFold(n_splits=n_folds).split(patients)):
            train_patients = patients[train_index]
            test_patients = patients[test_index]

            np.random.shuffle(train_patients)

            val_n = int(0.2 * len(train_patients))
            val_patients = train_patients[:val_n]
            train_patients = train_patients[val_n:]

            fold_dir = os.path.join(folds_dir, view, str(i_fold))

            _save_split(view_annotation, train_patients, ['No Finding'], os.path.join(fold_dir, 'normal', 'train'))
            _save_split(view_annotation, val_patients, ['No Finding'], os.path.join(fold_dir, 'normal', 'val'))
            _save_split(view_annotation, test_patients, ['No Finding'], os.path.join(fold_dir, 'normal', 'test'))
            _save_split(view_annotation, test_patients, valid_anomaly_labels, os.path.join(fold_dir, 'anomaly', 'test'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path_to_data_entry",
                        type=str,
                        default='./data/data/nih/Data_Entry_2017.csv',
                        help='path_to_data_entry')
    parser.add_argument("--path_to_train_val_list",
                        type=str,
                        default='./data/data/nih/train_val_list.txt',
                        help='path_to_train_val_list')
    parser.add_argument("-o", "--output_root",
                        required=True,
                        type=str)
    parser.add_argument("-n", "--n_folds",
                        type=int,
                        default=3,
                        help='n_folds')

    args = parser.parse_args()
    create_folds(args.path_to_data_entry, args.path_to_train_val_list, args.output_root, args.n_folds)
