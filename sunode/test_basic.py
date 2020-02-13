# type: ignore

import weakref

import numpy as np
from pytest import raises
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from sunode import basic, vector, matrix


VEC_TYPES = ["serial"]


@given(st.integers(), st.sampled_from(VEC_TYPES))
def test_empty(size, kind):
    try:
        if size >= 0:
            array = vector.empty_vector(size)
            assert array.shape == (size,)
        else:
            with raises(ValueError):
                vector.empty_vector(size)
    except (MemoryError, OverflowError):
        pass
    import gc

    gc.collect()


def test_vector_view():
    order = []

    def make_fin(name):
        def fin():
            order.append(name)

        return fin

    vector = vector.empty_vector(10)
    weakref.finalize(vector, make_fin("vector"))
    weakref.finalize(vector.c_ptr, make_fin("c_ptr"))

    view1 = vector.data
    weakref.finalize(view1, make_fin("view1"))

    view2 = view1[::2]
    weakref.finalize(view2, make_fin("view2"))

    view3 = vector.data
    weakref.finalize(view3, make_fin("view3"))

    del vector
    assert order == []
    del view1
    assert order == []
    del view2
    assert order == ["view2", "view1"]
    del view3
    assert order == ["view2", "view1", "view3", "vector", "c_ptr"]


def test_matrix_view():
    order = []

    def make_fin(name):
        def fin():
            order.append(name)

        return fin

    mat = matrix.empty_matrix((10, 8))
    weakref.finalize(mat, make_fin("vector"))
    weakref.finalize(mat.c_ptr, make_fin("c_ptr"))

    view1 = mat.data
    weakref.finalize(view1, make_fin("view1"))

    view2 = view1[0]
    weakref.finalize(view2, make_fin("view2"))

    view3 = mat.data
    weakref.finalize(view3, make_fin("view3"))

    del mat
    assert order == []
    del view1
    assert order == ["view1"]
    del view2
    assert order == ["view1", "view2"]
    del view3
    assert order == ["view1", "view2", "view3", "vector", "c_ptr"]


def test_sparse_view():
    order = []

    def make_fin(name):
        def fin():
            order.append(name)

        return fin

    mat = matrix.empty_matrix(
        (10, 8), "sparse", sparsity=np.random.randn(10, 8) > 0, format="csr"
    )
    weakref.finalize(mat, make_fin("vector"))
    weakref.finalize(mat.c_ptr, make_fin("c_ptr"))

    view1 = mat.data
    weakref.finalize(view1, make_fin("view1"))

    view2 = view1[::2]
    weakref.finalize(view2, make_fin("view2"))

    view3 = mat.data
    weakref.finalize(view3, make_fin("view3"))

    del mat
    assert order == []
    del view1
    assert order == []
    del view2
    assert order == ["view2", "view1"]
    del view3
    assert order == ["view2", "view1", "view3", "vector", "c_ptr"]


def test_sparse_realloc():
    mat = matrix.empty_matrix(
        (10, 8), "sparse", sparsity=np.random.randn(10, 8) > 0, format="csr"
    )
    data = mat.data
    with raises(RuntimeError):
        mat.realloc()

    del data
    mat.realloc()
    dat = mat.scipy

    with raises(RuntimeError):
        mat.realloc()
    assert mat.shape == (10, 8)


def test_from_numpy_ownership():
    order = []

    def make_fin(name):
        def fin():
            order.append(name)

        return fin

    array = np.zeros(10, dtype=basic.data_dtype)
    weakref.finalize(array, make_fin("array"))

    vec = vector.from_numpy(array, copy=False)

    weakref.finalize(vec, make_fin("vector"))
    weakref.finalize(vec.c_ptr, make_fin("c_ptr"))

    view1 = vec.data[::2]
    weakref.finalize(view1, make_fin("view1"))

    del array
    assert order == []
    del vec
    assert order == []
    del view1
    # The order ['array', 'c_ptr] is valid, because we call
    # the destructor of c_ptr in the destructor of vector.
    assert order == ['view1', 'vector', 'array', 'c_ptr']


def test_from_numpy():
    dtype = basic.data_dtype
    if dtype == np.float64:
        other_dtype = np.float32
    elif dtype == np.float32:
        other_dtype = np.float64
    else:
        assert False

    data = np.ones(10, dtype=dtype)
    vec = vector.from_numpy(data)
    assert len(vec) == len(data)
    assert vec.shape == data.shape
    assert vec.data.shape == data.shape
    assert np.all(data == vec.data)
    vec.data[:] = 2
    assert np.all(data == vec.data)
    assert np.all(data == 2)

    with raises(ValueError):
        vector.from_numpy(np.zeros((2, 2)))

    with raises(ValueError):
        vector.from_numpy(data.astype(other_dtype))

    with raises(ValueError):
        vector.from_numpy(np.zeros(()))

    vec = vector.from_numpy(np.zeros(0))
    assert vec.shape == (0,)
    assert vec.data.shape == (0,)
    assert len(vec) == 0


def test_empty_vector():
    vec = vector.empty_vector(10, kind="serial")
    assert vec.shape == (10,)
    assert vec.data.shape == (10,)

    with raises(ValueError):
        vector.empty_vector(-1)

    with raises(ValueError):
        vector.empty_vector(10, kind="foo")
