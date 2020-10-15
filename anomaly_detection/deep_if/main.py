import argparse
import yaml

from anomaly_detection.deep_if.train_evaluate import train_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['train_eval'])
    parser.add_argument('config', type=str, help='Config paths')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    train_evaluate(config)


if __name__ == '__main__':
    main()
