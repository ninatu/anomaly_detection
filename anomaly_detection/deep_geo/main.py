"""
Copyright (c) 2018 izikgo
Copyright (c) 2020 ninatu

This file is a modified file of the project: https://github.com/izikgo/AnomalyDetectionTransformations,
which was released under MIT License.
Go to https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/LICENSE for full license details.
"""

import argparse
import yaml

import tensorflow as tf
from keras.backend import set_session

from anomaly_detection.deep_geo.train import train
from anomaly_detection.deep_geo.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['train', 'eval', 'train_eval'])
    parser.add_argument('configs', type=str, nargs='*', help='Config paths')
    args = parser.parse_args()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    if args.action == 'train':
        assert len(args.configs) == 1
        train(_load_config(args.configs[0]))
    elif args.action == 'eval':
        assert len(args.configs) == 1
        evaluate(_load_config(args.configs[0]))
    else:
        assert len(args.configs) == 2
        train(_load_config(args.configs[0]))
        evaluate(_load_config(args.configs[1]))


def _load_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    main()
