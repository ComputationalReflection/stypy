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
    def random(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return float()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['size'], {
            'size': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'random', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if call_utilities.is_iterable(arguments[0]):
            inner_array = call_utilities.create_numpy_array_n_dimensions(numpy.float64(), call_utilities.get_dimensions(localization, arguments[0]))
            return call_utilities.create_numpy_array(inner_array)

        return call_utilities.create_numpy_array(numpy.float64())

    @staticmethod
    def rand(localization, proxy_obj, arguments):
        for i in range(len(arguments)):
            if not (Integer == type(arguments[i])):
                return StypyTypeError(localization, "Non-integer argument passed to rand function")

        dims = len(arguments)

        return call_utilities.create_numpy_array_n_dimensions(numpy.float64(), dims)
