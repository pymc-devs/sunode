from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Any

import numpy as np  # type: ignore

from pysundials_cffi import basic
from pysundials_cffi.builder import Option, Builder


@Builder._option
class state_vars(Option):
    def __call__(
        self, dims: Dict[str, Sequence[str]], coords: Dict[str, Sequence[Any]]
    ) -> None:
        data = self.builder._option_data
        data.state_dtype = np.dtype(vars)


@dataclass
class DTypes:
    n_states: int
    n_deriv: int
    state_dtype: np.dtype
    deriv_dtype: np.dtype
    extra_dtpe: np.dtype
    user_data: np.dtype

    def __init__(self, builder: Builder) -> None:
        options = builder._option_data
        build = builder._build_data

        assert build.y_template is not None
        self.n_states = len(build.y_template)

        states = options.state_dtype
        if states is None:
            dtype_states = np.dtype((basic.data_dtype, (self.n_states,)))
        else:
            pass
