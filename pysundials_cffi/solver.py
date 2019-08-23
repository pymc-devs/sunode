from __future__ import annotations

import dataclasses
from typing import Optional, Union, Any

import xarray  # type: ignore

from pysundials_cffi.basic import (
    Vector,
    DenseMatrix,
    SparseMatrix,
    Matrix,
    empty_vector,
    empty_matrix,
    LinearSolver,
)


class Solver:
    def __init__(self) -> None:
        pass

    def reset(self, data: Any = None, datafunc: Any = None) -> None:
        pass

    def step(self, target_time: float) -> None:
        pass

    def step_until_event(self) -> None:
        pass

    def step_once(self) -> None:
        pass

    def current_state(self) -> None:
        pass

    def current_sensitvity(self) -> None:
        pass

    def integrate(self) -> None:
        pass
