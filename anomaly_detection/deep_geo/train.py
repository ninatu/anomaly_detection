"""
Copyright (c) 2018 izikgo
Copyright (c) 2020 ninatu

This file is a modified file of the project: https://github.com/izikgo/AnomalyDetectionTransformations,
which was released under MIT License.
Go to https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/LICENSE for full license details.
"""

import os

import numpy as np
from keras.utils import to_categorical
import keras


from anomaly_detection.deep_geo.datasets import load_dataset
from anomaly_detection.deep_geo.networks import create_wide_residual_network
from anomaly_detection.deep_geo.transformations import Transformer
import tensorflow as tf
from keras.backend import set_session
from scipy.special import polygamma
from scipy.special._ufuncs import psi
import random
import json
import pickle
import tqdm


def train(config):
    print("Starting model training ...")
    if config['random_seed']:
        random_seed = config['random_seed']
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    checkpoint_root = config['checkpoint_root']
    os.makedirs(checkpoint_root, exist_ok=True)

    batch_size = config['batch_size']
    epochs = config['epochs']

    dataset_type = config['train_dataset']['dataset_type']
    transformer = _get_transformer(dataset_type)

    x_train, y_train = load_dataset(transformer, **config['train_dataset'])
    y_train_categorical = to_categorical(y_train)
    if config['val_dataset'] is not None:
        x_val, y_val = load_dataset(transformer, **config['val_dataset'])
        y_val_categorical = to_categorical(y_val)
        validation_data = x_val, y_val_categorical
    else:
        validation_data = None

    "=================================== training =================================="

    input_shape = x_train.shape[1:]
    model = _get_model(dataset_type, input_shape, transformer)
    model.compile('adam',
                'categorical_crossentropy',
                ['acc'])

    # callbacks

    callbacks = []

    log_path = os.path.join(checkpoint_root, 'log.csv')
    logger_callback = keras.callbacks.callbacks.CSVLogger(filename=log_path, append=False)
    callbacks.append(logger_callback)

    if config["checkpoint_callback"]:
        checkpoint_path = 'checkpoint_epoch_{epoch}.h5'
        checkpoint_path = os.path.join(checkpoint_root, checkpoint_path)

        callbacks.append(keras.callbacks.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            **config["checkpoint_callback"]
        ))

    if config['early_stopping_callback']:
        callbacks.append(keras.callbacks.callbacks.EarlyStopping(
            monitor='val_loss',
            **config['early_stopping_callback']
        ))

    history = model.fit(x=x_train, y=y_train_categorical, validation_data=validation_data,
                      batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)

    with open(os.path.join(checkpoint_root, 'history.json'), 'w') as f:
        json.dump(str(history.history), f)

    model.reset_metrics()

    checkpoint_path = os.path.join(checkpoint_root, 'checkpoint.h5')
    model.save(checkpoint_path)

    # save dirihlet parameters
    checkpoints = ['checkpoint.h5']
    if config["checkpoint_callback"]:
        for epoch in range(1, 1 + len(history.history['loss'])):
            checkpoint_name = 'checkpoint_epoch_{epoch}.h5'.format(epoch=epoch)
            if os.path.exists(os.path.join(checkpoint_root, checkpoint_name)):
                checkpoints.append(checkpoint_name)

    print("Saving dirichlet parameter of all checkpoints")
    for checkpoint_name in tqdm.tqdm(checkpoints):
        checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)
        model.load_weights(checkpoint_path)

        dirichlet_params = _compute_dirichlet_params(model, transformer, x_train, y_train, batch_size)
        dirichlet_params_path = 'dirichlet_params_{}.pickle'.format(os.path.splitext(checkpoint_name)[0])
        dirichlet_params_path = os.path.join(checkpoint_root, dirichlet_params_path)
        with open(dirichlet_params_path, 'wb') as f_out:
            pickle.dump(dirichlet_params, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model training is complete.")


def _get_transformer(dataset_type):
    if dataset_type in ['cats-vs-dogs', 'celeba', 'nih', 'camelyon16']:
        transformer = Transformer(16, 16)
    else:
        transformer = Transformer(8, 8)

    return transformer


def _get_model(dataset_type, input_shape, transformer):
    if dataset_type in ['cats-vs-dogs', 'celeba', 'nih', 'camelyon16']:
        n, k = (16, 8)
    else:
        n, k = (10, 4)

    model = create_wide_residual_network(input_shape, transformer.n_transforms, n, k)
    return model


def _compute_dirichlet_params(model, transformer, x_train, y_train, batch_size):
    mle_alpha_t = {}

    for t_ind in range(transformer.n_transforms):
        observed_data = x_train[y_train == t_ind]
        observed_dirichlet = model.predict(observed_data, batch_size=batch_size, verbose=True)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        alpha_sum_approx = _calc_approx_alpha_sum(observed_dirichlet)
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        mle_alpha_t[t_ind] = _fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
    return {'mle_alpha_t': mle_alpha_t}


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





