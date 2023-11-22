"""
In this file, we will define the containers for boundary conditions. My initial
idea is to used NamedTuples, but some kind of PyTree based approach might be
even nicer. Ultimately, it's important to make defining and unpacking boundary
conditions as easy intuitive as possible without performance degradation.

TODO: Possibly extend this with validation functions to check that the BCs are
      valid for the given mesh.
"""

from typing import Any, Callable, Optional, List, NamedTuple


class DirichletBCs(NamedTuple):
    location_fns: List[Callable]
    vecs: List[int]
    value_fns: List[Callable]


class NeumannBCs(NamedTuple):
    location_fns: List[Callable]
    value_fns: List[Callable]


class CauchyBCs(NamedTuple):
    # (Should this be renamed?)
    location_fns: List[Callable]
    value_fns: List[Callable]


class PeriodicBCs(NamedTuple):
    location_fns_A: List[Callable]
    location_fns_B: List[Callable]
    mappings: List[Callable]
    vecs: List[int]
