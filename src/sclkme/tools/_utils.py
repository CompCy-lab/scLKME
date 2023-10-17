from typing import Union

import numpy as np
from numpy.random import RandomState


def random_feats(
    X: np.ndarray,
    gamma: Union[int, float] = 1,
    D: int = 2000,
    random_state: Union[None, int, RandomState] = None,
):
    """
    Computes random fourier frequency features: https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Parameters
    X: np.ndarray
        array of input data (dimensions = cells x features)
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution
    D: int (default = 2000):
        dimensionality of the random Fourier frequency features, D/2 sin and D/2 cos basis
    frequency_seed: int (default = None):
        random state parameter
    ----------
    Returns
    phi: np.ndarray
        random Fourier frequency features (dimensions = cells x D)
    ----------
    """
    scale = 1 / gamma
    d = int(np.floor(D / 2))

    if isinstance(random_state, RandomState):
        W = random_state.normal(scale=scale, size=(X.shape[1], d))
    else:
        # set global random seed if the input is an integer
        np.random.seed(random_state)
        W = np.random.normal(scale=scale, size=(X.shape[1], d))

    XW = np.dot(X, W)
    sin_XW, cos_XW = np.sin(XW), np.cos(XW)
    phi = np.concatenate([cos_XW, sin_XW], axis=1)

    return phi
