#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, Number
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, wrap_type, \
    get_contained_elements_type
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.union_type import UnionType


class TypeModifiers:
    @staticmethod
    def det(localization, proxy_obj, arguments):
        param0 = call_utilities.get_inner_type(localization, arguments[0])
        dims = call_utilities.get_dimensions(localization, arguments[0])

        if dims == 1:
            return call_utilities.cast_to_numpy_type(param0)
        else:
            return call_utilities.create_numpy_array_n_dimensions(
                call_utilities.cast_to_numpy_type(param0),
                dims)