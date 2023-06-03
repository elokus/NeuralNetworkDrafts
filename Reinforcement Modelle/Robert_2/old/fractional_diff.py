import numpy as np


def frac_diff(x, d):
    """
    Fractionally difference time series

    :param x: numeric vector or univariate time series
    :param d: number specifying the fractional difference order.
    :return: fractionally differenced series
    """
    if np.isnan(np.sum(x)):
        return None

    n = len(x)
    if n < 2:
        return None

    x = np.subtract(x, np.mean(x))

    # calculate weights
    weights = [0] * n
    weights[0] = -d
    for k in range(2, n):
        weights[k - 1] = weights[k - 2] * (k - 1 - d) / k

    # difference series
    ydiff = list(x)

    for i in range(0, n):
        dat = x[:i]
        w = weights[:i]
        ydiff[i] = x[i] + np.dot(w, dat[::-1])

    return ydiff
