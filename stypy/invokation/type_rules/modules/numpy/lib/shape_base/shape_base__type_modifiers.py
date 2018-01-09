#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, DynamicType, Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type

class TypeModifiers:
    @staticmethod
    def tile(localization, proxy_obj, arguments):
        dims = 1
        if len(arguments) == 2 and call_utilities.is_iterable(arguments[1]):
            dims = call_utilities.get_dimensions(localization, arguments[1])

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array_n_dimensions(arguments[0], dims)

        return call_utilities.create_numpy_array_n_dimensions(get_contained_elements_type(localization, arguments[0]), dims)