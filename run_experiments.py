import subprocess
import argparse


MODELS = ['dpa', 'piad', 'deep_geo', 'deep_if']
DATASETS = ['cifar10', 'svhn', 'camelyon16', 'nih']
CLASSES = {
    'cifar10': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'svhn': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'camelyon16': ['healthy'],
    'nih': ['AP', 'PA', 'subset'],
}

EXPS = {
    'dpa': ['wo_pg', 'pg'],
    'dpa_ablation': ['wo_pg', 'pg', 'wo_pg_L1_unsupervised', 'wo_pg_unsupervised',
            'wo_pg_with_L1/weight_1.000', 'wo_pg_with_L1/weight_0.100',
            'wo_pg_with_adversarial/weight_1.000', 'wo_pg_with_adversarial/weight_0.100',
            'wo_pg_with_adversarial_L1/weight_1.000', 'wo_pg_with_adversarial_L1/weight_0.100'],
    'piad': ['with_cv', 'reproduce'],
    'deep_geo': ['with_cv', 'reproduce'],
    'deep_if': ['with_cv', 'reproduce'],
}

RUNS = ['0', '1', '2']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models",
                        type=str,
                        nargs='*',
                        default=MODELS,
                        help='models')
    parser.add_argument("--exps",
                        type=str,
                        nargs='*',
                        default=None,
                        help='exps')
    parser.add_argument("--datasets",
                        type=str,
                        nargs='*',
                        default=DATASETS,
                        help='datasets')
    parser.add_argument("--classes",
                        type=str,
                        nargs='*',
                        default=None,
                        help='classes')
    parser.add_argument("--runs",
                        type=str,
                        nargs='*',
                        default=RUNS,
                        help='runs')
    parser.add_argument("--ablation", action='store_true')
    args = parser.parse_args()

    models = args.models
    datasets = args.datasets
    runs = args.runs

    for model in args.models:
        for dataset in args.datasets:
            exps = args.exps
            if exps is None:
                exps = EXPS[model]
                if model == 'dpa' and dataset in ['camelyon16', 'nih'] and args.ablation:
                    exps = EXPS['dpa_ablation']

            for exp in exps:
                classes = CLASSES[dataset] if args.classes is None else args.classes

                for _class in classes:
                    for run in args.runs:

                        config = f"configs/{model}/{dataset}/final/{exp}/class_{_class}/run_{run}/train_eval.yaml"

                        subprocess_input = [
                            "python",
                            f"anomaly_detection/{model}/main.py",
                            "train_eval",
                            config
                        ]

                        if model != 'deep_if':
                            subprocess_input += [config]

                        print(subprocess_input)
                        subprocess.run(subprocess_input)
