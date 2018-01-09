#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import IterableDataStructure, Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, wrap_type, set_contained_elements_type
from stypy.types.union_type import UnionType
import numpy

class TypeModifiers:
    @staticmethod
    def ediff1d(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['to_end', 'to_begin'], {
            'to_end': IterableDataStructure,
            'to_begin': IterableDataStructure,
        }, 'ediff1d')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0])
        else:
            return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def unique(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['return_index', 'return_inverse', 'return_counts'], {
            'return_index': bool,
            'return_inverse': bool,
            'return_counts': bool,
        }, 'unique')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            ret_arr = call_utilities.create_numpy_array(arguments[0])
        else:
            ret_arr = call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

        if len(dvar.keys()) == 0:
            return ret_arr

        tup = wrap_type(tuple())
        union = UnionType.add(ret_arr, call_utilities.create_numpy_array(numpy.int32()))

        if len(dvar.keys()) == 1:
            set_contained_elements_type(localization, tup, union)
        if len(dvar.keys()) == 2:
            union = UnionType.add(union, call_utilities.create_numpy_array(numpy.int32()))
            set_contained_elements_type(localization, tup, union)
        if len(dvar.keys()) == 3:
            union = UnionType.add(union, call_utilities.create_numpy_array(numpy.int32()))
            union = UnionType.add(union, call_utilities.create_numpy_array(numpy.int32()))
            set_contained_elements_type(localization, tup, union)

        return tup