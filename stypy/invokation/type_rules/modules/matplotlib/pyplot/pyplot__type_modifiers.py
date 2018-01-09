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
    pass

 #   @staticmethod
 #   def jv(localization, proxy_obj, arguments):
        #
        # dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'out'], {
        #     'axis': Integer,
        #     'out': numpy.ndarray,
        # }, 'argmin', 0)
        #
        # if isinstance(dvar, StypyTypeError):
        #     return dvar
        #
        # if 'out' in dvar.keys():
        #     set_contained_elements_type(localization, dvar['out'], numpy.int32())
        #     return dvar['out']

  #      return call_utilities.create_numpy_array(numpy.float64())

