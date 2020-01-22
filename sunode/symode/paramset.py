import numba
import numpy as np
from collections import namedtuple
import sympy
import dataclasses
import pandas as pd

from typing import List, Tuple, Dict


def as_flattened(vals, base=None):
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


def as_nested(vals):
    out = {}

    current = {}
    current_name = None
    for (name, *names), val in vals.items():
        if not names:
            assert name not in out
            if current_name is not None:
                out[current_name] = as_nested(current)
                current_name = None
                current = {}
            out[name] = val
            continue

        if current_name is None:
            current_name = name
            current = {}

        if name == current_name:
            current[tuple(names)] = val
        else:
            out[name] = as_nested(current)
            current = {}
            current_name = name
    if current:
        out[current_name] = as_nested(current)
    return out
        

def count_items(dtype):
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


class ParamSet:
    pass


class DTypeSubset:
    dtype: np.dtype
    subset_dtype: np.dtype
    subset_view_dtype: np.dtype

    coords: Dict[str, pd.Index]
    dims: Dict[str, Tuple[str]]

    paths: List[Tuple[str]]
    subset_paths: List[Tuple[str]]

    # Map each path to a slice into the combined array
    flat_slices: Dict[Tuple[str], slice]
    flat_shapes: Dict[Tuple[str], slice]

    item_count: int

    def __init__(self, dims, subset_paths, fixed_dtype=None, coords=None, dim_basename=''):
        if coords is None:
            coords = {}
        else:
            coords = {name: pd.Index(coord) for name, coord in coords.items()}

        dtype = []
        subset_dtype = []
        subset_view_dtype = []

        paths = []

        slices = {}
        subset_slices = {}

        flat_slices = {}
        flat_shapes = {}

        dims_out = {}

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

    @property
    def n_subset(self):
        return count_items(self.subset_dtype)

    @property
    def n_items(self):
        return count_items(self.dtype)

    def set_from_subset(self, value_buffer, subset_buffer):
        value_buffer.view(self.subset_dtype).fill(subset_buffer)

    def as_dataclass(self, dataclass_name, flat_subset, flat_remainder, item_map=None):
        if item_map is None:
            item_map = lambda x: x

        def _as_dataclass(dataclass_name, dtype, subset_paths, flat_subset, flat_remainder):
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
                        name, subdtype, sub_paths, flat_subset, flat_remainder)
                fields.append((name, item))
            
            Type = dataclasses.make_dataclass(dataclass_name, [name for name, _ in fields])
            return Type(*[item for _, item in fields]), flat_subset, flat_remainder

        params, flat_subset, flat_remainder = _as_dataclass(
            dataclass_name, self.dtype, self.subset_paths, flat_subset, flat_remainder)
        assert len(flat_subset) == 0
        assert len(flat_remainder) == 0
        return params

    def from_dict(self, vals, out=None):
        pass

    def subset_from_dict(self, vals, out=None):
        pass

    def as_dict(self, vals):
        pass

    def subset_as_dict(self, vals):
        pass

    def slice_as_dataclass(self, array, array_subset, wrap_func=None):
        pass

    def remainder(self):
        remainder = list(set(self.paths) - set(self.subset_paths))
        return DTypeSubset(self.dims, remainder, coords=self.coords)


"""
class ParamSet:
    variables
    changeables
    dtype
    changeables_dtype
    changeables_view_dtype
    fixed_dtype
    fixed_view_dtype

    def as_dict(self, *, array=None, record=None):
        pass

    def changeable_as_dict(self, *, array=None, record=None):
        pass

    def fixed_as_dict(self, *, array=None, record=None):
        pass



    def array_from_dict(self, values, out=None):
        pass

    def record_from_dict(self, values, out=None):
        pass




    def __init__(self, variables, changeable_vars, fixed_dtype=None):
        self._variables = variables
        self._changeables = changeable_vars

        self._spec = spec
        self._dtype, self._idxs, self._count, self._paths, self._changeable_paths = ParamSet._parse_param_spec(spec)
        self._values = np.empty((), dtype=self._dtype)
        self.record = self._values.view(np.rec.recarray).reshape(1)[0]
        self._array = self._values.view((np.float64, len(self)))
        values = self.update_from_dict(initial_values, allow_missing=False)
    
    def __len__(self):
        return self._coun
    
    def copy(self):
        return ParamSet(self._spec, self.to_dict())
    
    @staticmethod
    def _parse_param_spec(spec):
        changeable_idxs = []
        dtype = []
        paths = []
        changeable_paths = []

        i = 0
        for key, value, *shape in spec:
            if isinstance(value, list):
                sub_dtype, sub_idxs, count, sub_paths, sub_changeable_paths = ParamSet._parse_param_spec(value)
                dtype.append((key, sub_dtype))
                changeable_idxs.extend(idx + i for idx in sub_idxs)
                paths.extend('.'.join([key, p]) for p in sub_paths)
                changeable_paths.extend('.'.join([key, p]) for p in sub_changeable_paths)
                i += count
            else:
                assert len(shape) <= 1
                if not shape:
                    shape = ()
                else:
                    shape = shape[0]
                dtype.append((key, np.float64, shape))
                count = np.product(shape, dtype=int)
                paths.append(key)
                if value:
                    changeable_idxs.extend(idx + i for idx in range(count))
                    changeable_paths.append(key)
                i += count
        return np.dtype(dtype), np.array(changeable_idxs, dtype=int), i, paths, changeable_paths
    
    @staticmethod
    def _update_from_dict(values, allow_missing, array, dtype):
        if not allow_missing:
            missing = set(dtype.fields) - set(values)
            if missing:
                raise ValueError('Missing parameters: %s' % missing)
        for key, val in values.items():
            if key not in dtype.fields:
                raise KeyError('Unknown parameter: %s' % key)
            subdtype, offset = dtype.fields[key]
            offset = offset // 8
            if isinstance(val, dict):
                ParamSet._update_from_dict(
                    val, allow_missing, array[offset:], subdtype)
            else:
                data = np.asarray(val)
                if np.dtype((data.dtype, data.shape)) != subdtype:
                    raise ValueError('Invalid dtype %s. Should be %s'
                                     % (data.dtype, subdtype))
                array[offset:data.size + offset] = data.ravel()

    def update_from_dict(self, values, allow_missing=True):
        ParamSet._update_from_dict(
            values, allow_missing, self._array, self._dtype)
        
    def to_dict(self):
        def to_dict_rec(value):
            dtype = value.dtype
            out = {}

            for key, (subdtype, _) in dtype.fields.items():
                if not subdtype.fields:
                    val = value[key].copy()
                else:
                    val = to_dict_rec(value[key])
                out[key] = val
            return out
        return to_dict_rec(self.record)

    def pprint(self):
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        json_dump = json.dumps(self.to_dict(), cls=NumpyEncoder, indent=4)
        print(json_dump)
        
    def print_changeable(self):
        import pprint
        def _get_names_changeable(name_list,name_upper,params):
            for param in params:
                if isinstance(param[1], bool):
                    if param[1]:
                        name_list.append(name_upper + param[0])
                else:
                    _get_names_changeable(name_list,name_upper + param[0] + '.', param[1])
        name_list = []
        _get_names_changeable(name_list,'',self._spec)

        pprint.pprint(dict(zip(name_list,self.changeable_array())))
    
    def array_view(self):
        return self._array

    def changeable_array(self):
        return self._array[self._idxs]
    
    def fixed_array(self):
        return np.delete(self._array, self._idxs)
    
    def set_changeable(self, values):
        self._array[self._idxs] = values

    def set_fixed(self, values):
        all_idxs = range(len(self._array))
        idxs = [i for i in all_idxs if i not in self._idxs]
        assert values.shape == (len(idxs),)
        self._array[idxs] = values

    @property
    def paths(self):
        return self._paths
    
    @property
    def changeable_paths(self):
        return self._changeable_paths
    
    def as_sympy(self, array_symbol):
        return self._as_sympy('SympyParams', array_symbol, self._dtype)
    
    def _as_sympy(self, name, array_symbol, dtype):
        fields = dtype.fields
        if fields is None:
            print(dtype)
            return array_symbol[0, 0]
        Type = namedtuple(name, list(fields))
        print(fields, array_symbol, array_symbol.shape)
        vals = [self._as_sympy(name, array_symbol[:, offset//8:], subdtype)
                for name, (subdtype, offset) in fields.items()]
        return Type(*vals)
    
    def as_sympy(self, name):
        fixed = sympy.MatrixSymbol(
            name + 'fixed', 1, len(self) - len(self._idxs))
        changeable = sympy.MatrixSymbol(
            name + 'changeable', 1, len(self._idxs))
        #fixed = sympy.symarray(name + 'fixed', (1, len(self) - len(self._idxs)))
        #changeable = sympy.symarray(name + 'changeable', (1, len(self._idxs)))
        return (
            ParamSet._as_sympy('SympyParams', self._spec, fixed, changeable)[0],
            fixed,
            changeable
        )
    
    def slice_by_path(self, path):
        return self._slice_by_path(path.split('.'), self.record.dtype)
    
    def _slice_by_path(self, path, record):
        item, *path = path
        rec, offset = record.fields[item]
        offset = offset // 8
        if path:
            inner = self._slice_by_path(path, rec)
            return slice(inner.start + offset, inner.stop + offset)
        size = rec.itemsize // 8
        return slice(offset, offset + size)
    
    def changeable_idxs_by_path(self, path):
        idxs = self.slice_by_path(path)
        start, stop, step = idxs.indices(len(self.array_view()))
        idxs = list(self._idxs)
        try:
            return [idxs.index(i) for i in range(start, stop, step)]
        except ValueError:
            raise KeyError('Path is not changeable: %s' % path)
    
    @staticmethod
    def _as_sympy(name, spec, fixed, changeable):
        i_fixed = 0
        i_changeable = 0
        
        fields = []
        
        for key, value, *shape in spec:
            if isinstance(value, list):
                if changeable is None or changeable.shape[-1] == i_changeable:
                    rem_changeable = None
                else:
                    rem_changeable = changeable[:, i_changeable:]
                if fixed is None or fixed.shape[-1] == i_fixed:
                    rem_fixed = None
                else:
                    rem_fixed = fixed[:, i_fixed:]
                params, inner_i_fixed, inner_i_changeable = ParamSet._as_sympy(
                    key, value, rem_fixed, rem_changeable)
                i_fixed += inner_i_fixed
                i_changeable += inner_i_changeable
            else:
                assert len(shape) <= 1
                if not shape:
                    shape = ()
                else:
                    shape = shape[0]
                count = np.product(shape, dtype=int)
                if value:
                    if shape == ():
                        params = changeable[0, i_changeable]
                    else:
                        vals = [changeable[0, i]
                                for i in range(i_changeable, i_changeable+count)]
                        params = sympy.Array(vals).reshape(*shape)
                    i_changeable += count
                else:
                    if shape == ():
                        params = fixed[0, i_fixed]
                    else:
                        vals = [fixed[0, i] for i in range(i_fixed, i_fixed+count)]
                        params = sympy.Array(vals).reshape(*shape)
                    i_fixed += count
            fields.append((key, params))
        
        Type = dataclasses.make_dataclass(name, [name for name, _ in fields])
        return Type(*[params for _, params in fields]), i_fixed, i_changeable


    @staticmethod
    def _contains_param(path, record):
        split_path = path.split(".", 1)
        if len(split_path) == 1:
            return split_path[0] in record.dtype.fields
        else:
            item, path = split_path
            if item not in record.dtype.fields:
                return False
            return ParamSet._contains_param(path, record[item])


    def contains_param(self, param):
        return ParamSet._contains_param(param, self.record)


    def is_changeable(self, path):
        return self.slice_by_path(path).start in self._idxs
"""
