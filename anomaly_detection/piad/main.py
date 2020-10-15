import argparse
import yaml
import time

from anomaly_detection.piad.train import Trainer
from anomaly_detection.piad.evaluate import evaluate


def _load_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['train', 'eval', 'train_eval'])
    parser.add_argument('configs', type=str, nargs='*', help='Config paths')
    args = parser.parse_args()

    if args.action == 'train':
        assert len(args.configs) == 1
        start_time = time.time()
        trainer = Trainer(_load_config(args.configs[0]))
        trainer.train()
        print(f"Training is complete. Took: {(time.time() - start_time) / 60:.02f}m")
    elif args.action == 'eval':
        assert len(args.configs) == 1
        start_time = time.time()
        evaluate(_load_config(args.configs[0]))
        print(f"Evaluation is complete. Took: {(time.time() - start_time) / 60:.02f}m")
    else:
        assert len(args.configs) == 2

        start_time = time.time()
        trainer = Trainer(_load_config(args.configs[0]))
        trainer.train()
        print(f"Training is complete. Took: {(time.time() - start_time) / 60:.02f}m")
        del trainer

        start_time = time.time()
        evaluate(_load_config(args.configs[1]))
        print(f"Evaluation is complete. Took: {(time.time() - start_time) / 60:.02f}m")


if __name__ == '__main__':
    main()
