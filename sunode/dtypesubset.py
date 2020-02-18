import numpy as np
from collections import namedtuple
import dataclasses

import pandas as pd

from typing import List, Tuple, Dict, Union, Any, Optional, Callable


def as_flattened(vals: Dict[str, Any], base: Optional[Tuple[str, ...]] = None) -> Dict[Tuple[str, ...], Any]:
    if base is None:
        base = tuple()
    out = {}
    for name, val in vals.items():
        if isinstance(val, dict):
            flat = as_flattened(val, base=base + (name,))
            out.update(flat)
        else:
            out[base + (name,)] = val
    return out


def as_nested(vals: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for names, val in vals.items():
        assert len(names) >= 1
        current = out
        for name in names[:-1]:
            current = current.setdefault(name, {})
        assert names[-1] not in current
        current[names[-1]] = val
    return out


def count_items(dtype: np.dtype) -> int:
    if dtype.fields is None:
        prod = 1
        for length in dtype.shape:
            prod *= length
        return prod
    else:
        num = 0
        for dt, _ in dtype.fields.values():
            num += count_items(dt)
        return num


def _as_dict(data: np.ndarray) -> Dict[str, Any]:
    if data.dtype.fields is not None:
        return {name: _as_dict(data[name]) for name in data.dtype.fields}
    else:
        return data


def _from_dict(data: np.ndarray, vals: Dict[str, Any]) -> None:
    if data.dtype.fields is not None:
        for name, (subtype, _) in data.dtype.fields.items():
            if subtype.fields is not None:
                _from_dict(data[name], vals[name])
            else:
                data[name] = vals[name]
    else:
        data[...] = vals


Shape = Tuple[int, ...]
Path = Tuple[str, ...]


class DTypeSubset:
    dtype: np.dtype
    subset_dtype: np.dtype
    subset_view_dtype: np.dtype

    coords: Dict[str, pd.Index]
    dims: Dict[str, Any]

    paths: List[Path]
    subset_paths: List[Path]

    # Map each path to a slice into the combined array
    flat_slices: Dict[Path, slice]
    flat_shapes: Dict[Path, Shape]

    item_count: int

    _remainder: Optional['DTypeSubset']

    def __init__(
        self,
        dims: Dict[str, Any],
        subset_paths: List[Path],
        fixed_dtype: Optional[np.dtype] = None,
        coords: Optional[Dict[str, pd.Index]] = None,
        dim_basename: str = ''
    ) -> None:
        if coords is None:
            coords = {}
        else:
            coords = {name: pd.Index(coord) for name, coord in coords.items()}

        dtype: List[Tuple[str, Any, Tuple[int, ...]]] = []
        subset_dtype: List[Tuple[str, Any, Tuple[int, ...]]] = []
        subset_view_dtype = []

        paths: List[Tuple[str, ...]] = []

        flat_slices: Dict[Tuple[str, ...], slice] = {}
        flat_shapes: Dict[Tuple[str, ...], Tuple[int, ...]] = {}

        dims_out: Dict[str, Any] = {}

        subset_names = []
        subset_offsets = []
        offset = 0
        item_count = 0
        for name, val in dims.items():
            if isinstance(val, dict):
                flat_sub_paths = [p[1:] for p in subset_paths if len(p) > 0 and p[0] == name]
                sub_subset = DTypeSubset(val, flat_sub_paths, fixed_dtype=fixed_dtype, coords=coords)
                coords.update(sub_subset.coords)
                dtype.append((name, sub_subset.dtype, ()))
                if sub_subset.subset_dtype.itemsize > 0:
                    subset_dtype.append((name, sub_subset.subset_dtype, ()))
                    subset_view_dtype.append(sub_subset.subset_view_dtype)
                    subset_names.append(name)
                    subset_offsets.append(offset)

                paths.extend((name,) + path for path in sub_subset.paths)
                dims_out[name] = sub_subset.dims
                for path in sub_subset.paths:
                    full_path = (name,) + path
                    assert full_path not in flat_slices and full_path not in flat_shapes
                    sub_slice = sub_subset.flat_slices[path]
                    flat_slices[full_path] = slice(
                        sub_slice.start + item_count,
                        sub_slice.stop + item_count,
                    )
                    flat_shapes[full_path] = sub_subset.flat_shapes[path]
                item_count += sub_subset.item_count
            else:
                if fixed_dtype is None:
                    val_dtype, val = val
                else:
                    val_dtype = fixed_dtype
                if isinstance(val, (int, str)):
                    val = (val,)
                shape = []
                item_dims = []
                for i, dim in enumerate(val):
                    if isinstance(dim, str):
                        if dim not in coords:
                            raise KeyError('Unknown dimension name: %s' % dim)
                        length = len(coords[dim])
                        dim_name = dim
                    else:
                        length = dim
                        index = pd.RangeIndex(length, name='%s_%s_dim%s__' % (dim_basename, name, i))
                        dim_name = index.name
                        assert dim_name not in coords
                        coords[dim_name] = index
                    item_dims.append(dim_name)
                    shape.append(length)
                dims_out[name] = (val_dtype, item_dims)
                dtype.append((name, val_dtype, tuple(shape)))
                if (name,) in subset_paths:
                    subset_dtype.append((name, val_dtype, tuple(shape)))
                    subset_view_dtype.append((val_dtype, tuple(shape)))
                    subset_offsets.append(offset)
                    subset_names.append(name)
                paths.append((name,))
                length = 1
                for dim_len in shape:
                    length *= dim_len
                flat_slices[(name,)] = slice(item_count, item_count + length)
                flat_shapes[(name,)] = tuple(shape)
                item_count += length
            offset += np.dtype([dtype[-1]]).itemsize
        self.dtype = np.dtype(dtype)
        self.subset_dtype = np.dtype(subset_dtype)
        self.subset_view_dtype = np.dtype({
            'names': subset_names,
            'formats': subset_view_dtype,
            'offsets': subset_offsets,
            'itemsize': self.dtype.itemsize,
        })

        self.item_count = item_count
        self.flat_shapes = flat_shapes
        self.flat_slices = flat_slices

        self.coords = coords
        self.paths = paths
        self.dims = dims_out

        # Make sure the order of subset_paths is correct
        self.subset_paths = [path for path in paths if path in subset_paths]
        self._remainder = None

    @property
    def n_subset(self) -> int:
        return count_items(self.subset_dtype)

    @property
    def n_items(self) -> int:
        return count_items(self.dtype)

    def set_from_subset(self, value_buffer: np.ndarray, subset_buffer: np.ndarray) -> None:
        value_buffer.view(self.subset_dtype).fill(subset_buffer)

    def as_dataclass(
        self,
        dataclass_name: str,
        flat_subset: np.ndarray,
        flat_remainder: np.ndarray,
        item_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Any:
        if item_map is None:
            item_map = lambda x: x

        def _as_dataclass(
            dataclass_name: str,
            dtype: np.dtype,
            subset_paths: List[Path],
            flat_subset: np.ndarray,
            flat_remainder: np.ndarray,
            item_map: Callable[[np.ndarray], np.ndarray],
        ) -> Any:
            fields = []
            
            for name, (subdtype, _) in dtype.fields.items():
                if subdtype.fields is None:
                    count = count_items(subdtype)
                    if (name,) in subset_paths:
                        assert len(flat_subset) >= count
                        item = item_map(flat_subset[:count].reshape(subdtype.shape))
                        flat_subset = flat_subset[count:]
                    else:
                        assert len(flat_remainder) >= count
                        item = item_map(flat_remainder[:count].reshape(subdtype.shape))
                        flat_remainder = flat_remainder[count:]
                else:
                    sub_paths = [p[1:] for p in subset_paths if len(p) > 0 and p[0] == name]
                    item, flat_subset, flat_remainder = _as_dataclass(
                        name, subdtype, sub_paths, flat_subset, flat_remainder, item_map)
                fields.append((name, item))
            
            Type = dataclasses.make_dataclass(dataclass_name, [name for name, _ in fields])
            return Type(*[item for _, item in fields]), flat_subset, flat_remainder

        params, flat_subset, flat_remainder = _as_dataclass(
            dataclass_name, self.dtype, self.subset_paths, flat_subset, flat_remainder, item_map)
        assert len(flat_subset) == 0
        assert len(flat_remainder) == 0
        return params

    def from_dict(self, vals: Dict[str, Any], out: Optional[np.ndarray] = None) -> None:
        if out is None:
            out = np.zeros((1,), dtype=self.dtype)[0]
        _from_dict(out, vals)

    def subset_from_dict(self, vals: Dict[str, Any], out: Optional[np.ndarray] = None) -> None:
        if out is None:
            out = np.zeros((1,), dtype=self.subset_dtype)[0]
        _from_dict(out, vals)

    def as_dict(self, vals: np.ndarray) -> Dict[str, Any]:
        if vals.dtype != self.dtype:
            raise ValueError('Invalid dtype.')
        return _as_dict(vals)

    def subset_as_dict(self, vals: np.ndarray) -> Dict[str, Any]:
        if vals.dtype != self.subset_dtype:
            raise ValueError('Invalid dtype.')
        return _as_dict(vals)

    @property
    def remainder(self) -> 'DTypeSubset':
        if self._remainder is None:
            remainder = list(set(self.paths) - set(self.subset_paths))
            self._remainder = DTypeSubset(self.dims, remainder, coords=self.coords)
        return self._remainder
