"""
Copyright (c) 2018 izikgo
Copyright (c) 2020 ninatu

This file is a modified file of the project: https://github.com/izikgo/AnomalyDetectionTransformations,
which was released under MIT License.
Go to https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/LICENSE for full license details.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend import set_session
from sklearn.metrics import roc_curve, auc
from scipy.special import polygamma
from scipy.special._ufuncs import psi
import pickle

from anomaly_detection.deep_geo.datasets import load_dataset
from anomaly_detection.deep_geo.train import _get_transformer, _get_model


def evaluate(config):
    print("Starting model evaluation ...")

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    checkpoint_path = config['test_checkpoint_path']
    test_batch_size = config['test_batch_size']
    results_root = config['results_root']
    os.makedirs(results_root, exist_ok=True)

    checkpoint_name = os.path.basename(checkpoint_path)
    dirichlet_params_path = 'dirichlet_params_{}.pickle'.format(os.path.splitext(checkpoint_name)[0])
    dirichlet_params_path = os.path.join(os.path.dirname(checkpoint_path), dirichlet_params_path)
    with open(dirichlet_params_path, 'rb') as f_in:
        dirichlet_params = pickle.load(f_in)
    mle_alpha_t = dirichlet_params['mle_alpha_t']

    # compute scores for normal
    dataset_type = config['test_datasets']['normal']['dataset_type']
    transformer = _get_transformer(dataset_type)

    x_test_normal, y_test_normal = load_dataset(transformer, **config['test_datasets']['normal'])
    scores_normal = np.zeros(int(len(x_test_normal) / transformer.n_transforms))

    input_shape = x_test_normal.shape[1:]
    model = _get_model(dataset_type, input_shape, transformer)
    model.load_weights(checkpoint_path)

    for t_ind in range(transformer.n_transforms):
        p_normal = model.predict(x_test_normal[y_test_normal == t_ind], batch_size=test_batch_size, verbose=True)
        scores_normal += _dirichlet_normality_score(mle_alpha_t[t_ind], p_normal)
    scores_normal /= transformer.n_transforms
    del x_test_normal, y_test_normal

    # compute scores for anomaly
    x_test_anomaly, y_test_anomaly = load_dataset(transformer, **config['test_datasets']['anomaly'])
    scores_anomaly = np.zeros(int(len(x_test_anomaly) / transformer.n_transforms))

    for t_ind in range(transformer.n_transforms):
        p_anomaly = model.predict(x_test_anomaly[y_test_anomaly == t_ind], batch_size=test_batch_size, verbose=True)
        scores_anomaly += _dirichlet_normality_score(mle_alpha_t[t_ind], p_anomaly)
    scores_anomaly /= transformer.n_transforms
    del x_test_anomaly, y_test_anomaly

    # compute roc auc
    labels = np.concatenate((np.ones(scores_normal.shape), np.zeros(scores_anomaly.shape)))
    scores = np.concatenate((scores_normal, scores_anomaly))

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    results = pd.DataFrame([[roc_auc]], columns=['ROC AUC'])

    print("Model evaluation is complete. Results: ")
    print(results)
    results.to_csv(os.path.join(results_root, 'results.csv'))


def _calc_approx_alpha_sum(observations):
    N = len(observations)
    f = np.mean(observations, axis=0)

    return (N * (len(f) - 1) * (-psi(1))) / (
            N * np.sum(f * np.log(f)) - np.sum(f * np.sum(np.log(observations), axis=0)))


def _inv_psi(y, iters=5):
    # initial estimate
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

    for _ in range(iters):
        x = x - (psi(x) - y) / polygamma(1, x)
    return x


def _fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
    alpha_new = alpha_old = alpha_init
    for _ in range(max_iter):
        alpha_new = _inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
        if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
            break
        alpha_old = alpha_new
    return alpha_new


def _dirichlet_normality_score(alpha, p):
    return np.sum((alpha - 1) * np.log(p + 1e-6), axis=-1)
