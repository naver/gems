GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import numpy as np
import random

def sample_items(num_items, shape):
    """
    Randomly sample a number of items.
    Parameters
    ----------
    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    Returns
    -------
    items: np.array of shape [shape]
        Sampled item ids.
    """

    res_items = np.random.randint(0, num_items, shape, dtype=np.int64)
    return res_items
