from pysundials_cffi import basic

import numpy as np
from pytest import raises


def test_from_numpy():
    dtype = basic.data_dtype
    if dtype == np.float64:
        other_dtype = np.float32
    elif dtype == np.float32:
        other_dtype = np.float64
    else:
        assert False

    data = np.ones(10, dtype=dtype)
    vec = basic.from_numpy(data)
    assert len(vec) == len(data)
    assert vec.shape == data.shape
    assert vec.data.shape == data.shape
    assert np.all(data == vec.data)
    vec.data[:] = 2
    assert np.all(data == vec.data)
    assert np.all(data == 2)

    with raises(ValueError):
        basic.from_numpy(np.zeros((2, 2)))

    with raises(ValueError):
        basic.from_numpy(data.astype(other_dtype))

    with raises(ValueError):
        basic.from_numpy(np.zeros(()))

    vec = basic.from_numpy(np.zeros(0))
    assert vec.shape == (0,)
    assert vec.data.shape == (0,)
    assert len(vec) == 0


def test_empty_vector():
    vec = basic.empty_vector(10, kind="serial")
    assert vec.shape == (10,)
    assert vec.data.shape == (10,)

    with raises(ValueError):
        basic.empty_vector(-1)

    with raises(ValueError):
        basic.empty_vector(10, kind="foo")
