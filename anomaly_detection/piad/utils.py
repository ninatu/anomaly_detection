import re
from collections import defaultdict

import numpy as np


# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


class GradientNormHelper:
    def __init__(self, history_len=200, exp_smoothing_alpha=0.05):
        self._history_len = history_len
        self._alpha = exp_smoothing_alpha

        self._grad_norm_dict = \
            defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: []
                    )
            )

        )

    def update_grad_norm_dict(self, model, model_name, loss, loss_name):
        model.zero_grad()
        loss.backward(retain_graph=True)

        grad_norm = {}
        for layer_name, value in model.named_parameters():
            layer_name = layer_name.replace('model.', '').replace('.', '/')
            if re.match(r'.*weight$', layer_name) and value.grad is not None:
                grad_norm = np.linalg.norm(value.grad.data.cpu().numpy())
                self._grad_norm_dict[model_name][loss_name][layer_name].append(grad_norm)
        model.zero_grad()
        return grad_norm

    def get_loss_weight(self, model_name, base_loss_name):
        average_values = defaultdict(lambda: dict())
        grad_norm = self._grad_norm_dict[model_name]

        for loss_name, loss_dict in grad_norm.items():
            for layer, values in loss_dict.items():
                values = values[-self._history_len:]
                loss_dict[layer] = values

                average_values[layer][loss_name] = ewma_vectorized(values, self._alpha)

        ratio_values = defaultdict(lambda: [])

        for layer, layer_dict in average_values.items():
            base_values = layer_dict[base_loss_name]

            for loss_name, values in layer_dict.items():
                ratio_values[loss_name].extend((base_values / values)[-self._history_len:])

        average_ratio = dict()
        for loss_name, ratios in ratio_values.items():
            average_ratio[loss_name] = np.median(ratios)
        return average_ratio