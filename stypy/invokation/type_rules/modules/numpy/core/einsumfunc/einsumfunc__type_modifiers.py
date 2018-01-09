#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, Number, Str, RealNumber, IterableDataStructure, DynamicType
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type
from stypy.types.union_type import UnionType

class TypeModifiers:
    @staticmethod
    def einsum(localization, proxy_obj, arguments):
        if isinstance(arguments[-1], dict):
            if Str == type(arguments[0]):
                arg_num = 2
            else:
                arg_num = 1

            dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                           ['out', 'dtype', 'order', 'casting', 'optimize'], {
                                                               'out': IterableDataStructure,
                                                               'dtype': type,
                                                               'order': Str,
                                                               'casting': Str,
                                                               'optimize': [bool, Str]
                                                           }, 'einsum', arg_num)

            if isinstance(dvar, StypyTypeError):
                return dvar

            val_temp = call_utilities.check_possible_values(dvar, 'order', ['C', 'F', 'A', 'K'])
            if isinstance(val_temp, StypyTypeError):
                return val_temp

            val_temp = call_utilities.check_possible_values(dvar, 'casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
            if isinstance(val_temp, StypyTypeError):
                return val_temp

            val_temp = call_utilities.check_possible_values(dvar, 'optimize', ['greedy', 'optimal', False, True])
            if isinstance(val_temp, StypyTypeError):
                return val_temp

            arguments = arguments[:-1]
        else:
            dvar = dict()

        typ = None
        if Str == type(arguments[0]):
            arg_list = arguments[1:]
            if Number == type(arguments[1]) and 'out' in dvar:
                return dvar['out']
        else:
            arg_list = arguments

        for arg in arg_list:
            if call_utilities.is_iterable(arg):
                typ_temp = call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, arg))
                typ = call_utilities.cast_to_greater_numpy_type(typ, typ_temp)

        union = UnionType.add(typ, call_utilities.create_numpy_array(DynamicType))

        if 'out' in dvar:
            set_contained_elements_type(localization, dvar['out'], DynamicType)
            return call_utilities.create_numpy_array(DynamicType)

        return union


