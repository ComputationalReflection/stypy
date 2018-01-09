#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type


class TypeModifiers:
    @staticmethod
    def nansum(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out', 'keepdims'], {
            'axis': int,
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'nansum')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

        return call_utilities.cast_to_numpy_type(get_contained_elements_type(localization, arguments[0]))
