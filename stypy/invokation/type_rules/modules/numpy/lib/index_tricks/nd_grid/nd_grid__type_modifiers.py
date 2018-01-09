#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type
from stypy.types import union_type

class TypeModifiers:
    @staticmethod
    def __getitem__(localization, proxy_obj, arguments):
        if isinstance(arguments[0], tuple):
            num = len(arguments[0])
            typ = call_utilities.cast_to_numpy_type(arguments[0][0].get_wrapped_type().start)
            union = None

            for i in range(num):
                union = union_type.UnionType.add(union, call_utilities.create_numpy_array_n_dimensions(typ, num))

            tup = call_utilities.wrap_contained_type((union,))
            tup.set_contained_type(union)

            return tup

        return None

