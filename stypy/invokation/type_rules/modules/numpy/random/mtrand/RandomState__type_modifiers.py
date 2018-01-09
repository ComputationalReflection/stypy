#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, \
    IterableDataStructureWithTypedElements, Str, Number
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, get_contained_elements_type


# class TypeModifiers:
#     @staticmethod
#     def random_sample(localization, proxy_obj, arguments, func_name='random_sample'):
#         dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['size', 'out'], {
#             'out': [Number, numpy.ndarray],
#             'size': numpy.ndarray,
#         }, func_name)
#
#         if isinstance(dvar, StypyTypeError):
#             return dvar
#
#         if Number == type(arguments[0]):
#             ret = call_utilities.cast_to_numpy_type(numpy.float64())
#         else:
#             try:
#                 ret = call_utilities.create_numpy_array_n_dimensions(
#                     call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, arguments[0])),
#                  call_utilities.get_dimensions(localization, arguments[0]))
#
#             except Exception as ex:
#                 return StypyTypeError(localization, str(ex))
#
#         if 'out' in dvar.keys():
#             set_contained_elements_type(localization, dvar['out'],
#                                         ret)
#             return dvar['out']
#
#         return ret