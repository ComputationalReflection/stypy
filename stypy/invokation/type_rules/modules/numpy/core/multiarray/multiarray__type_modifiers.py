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
    def array(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0], False)

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['dtype',
            'copy',
            'order',
            'subok',
            'ndmin'], {
            'dtype': type,
            'copy': bool,
            'order': Str,
            'subok': bool,
            'ndmin': Integer,
        }, 'array')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar:
            dtype = dvar['dtype']
        else:
            dtype = None

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]), dtype)

    @staticmethod
    def empty(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(numpy.float64(), False)

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'order'], {
            'dtype': type,
            'order': Str,
        }, 'array')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar:
            dtype = dvar['dtype']
        else:
            dtype = None

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]), dtype)


    @staticmethod
    def arange(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['stop', 'step', 'dtype'], {
            'stop': RealNumber,
            'step': RealNumber,
            'dtype': type,
        }, 'arange')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar:
            dtype = dvar['dtype']
        else:
            dtype = None

        return call_utilities.create_numpy_array(arguments[0], dtype)

    @staticmethod
    def bincount(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['weights', 'minlength'], {
                                                           'weights': IterableDataStructure,
                                                           'minlength': Integer,
                                                       }, 'bincount')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return call_utilities.create_numpy_array(numpy.int32())

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

    @staticmethod
    def dot(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['out'], {
                                                           'out': numpy.ndarray,
                                                       }, 'dot')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return call_utilities.cast_to_greater_numpy_type(arguments[0], arguments[1])

        if Number == type(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            c_t = get_contained_elements_type(localization, arguments[1])
            if not 'out' in dvar.keys():
                return call_utilities.create_numpy_array(c_t)
            else:
                set_contained_elements_type(localization, dvar['out'], c_t)
                return dvar['out']

        if Number == type(arguments[1]) and call_utilities.is_iterable(arguments[0]):
            c_t = get_contained_elements_type(localization, arguments[0])
            if not 'out' in dvar.keys():
                return call_utilities.create_numpy_array(c_t)
            else:
                set_contained_elements_type(localization, dvar['out'], c_t)
                return dvar['out']

        if call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            if call_utilities.get_dimensions(localization, arguments[0]) == 1 and call_utilities.get_dimensions(localization, arguments[1]) == 1:
                return call_utilities.cast_to_greater_numpy_type(call_utilities.get_inner_type(localization, arguments[0]), call_utilities.get_inner_type(localization, arguments[1]))

            typ = call_utilities.cast_to_greater_numpy_type(call_utilities.get_inner_type(localization, arguments[0]), call_utilities.get_inner_type(localization, arguments[1]))
            for i in range (call_utilities.get_dimensions(localization, arguments[0])):
                typ = call_utilities.create_numpy_array(typ)

            if not 'out' in dvar.keys():
                return typ
            else:
                set_contained_elements_type(localization, dvar['out'], get_contained_elements_type(localization, typ))
                return dvar['out']

        return arguments[0]

    @staticmethod
    def inner(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['out'], {
                                                           'out': numpy.ndarray,
                                                       }, 'inner')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return call_utilities.cast_to_greater_numpy_type(arguments[0], arguments[1])

        if Number == type(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            c_t = get_contained_elements_type(localization, arguments[1])
            if not 'out' in dvar.keys():
                return call_utilities.create_numpy_array(c_t)
            else:
                set_contained_elements_type(localization, dvar['out'], c_t)
                return dvar['out']

        if Number == type(arguments[1]) and call_utilities.is_iterable(arguments[0]):
            c_t = get_contained_elements_type(localization, arguments[0])
            if not 'out' in dvar.keys():
                return call_utilities.create_numpy_array(c_t)
            else:
                set_contained_elements_type(localization, dvar['out'], c_t)
                return dvar['out']

        if call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            if call_utilities.get_dimensions(localization, arguments[0]) == 1 and call_utilities.get_dimensions(localization, arguments[1]) == 1:
                return call_utilities.cast_to_greater_numpy_type(call_utilities.get_inner_type(localization, arguments[0]), call_utilities.get_inner_type(localization, arguments[1]))

            # typ = call_utilities.cast_to_greater_numpy_type(call_utilities.get_inner_type(localization, arguments[0]), call_utilities.get_inner_type(localization, arguments[1]))
            # for i in range (call_utilities.get_dimensions(localization, arguments[0])):
            #     typ = call_utilities.create_numpy_array(typ)

            typ = call_utilities.create_numpy_array(DynamicType())
            if not 'out' in dvar.keys():
                return typ
            else:
                set_contained_elements_type(localization, dvar['out'], get_contained_elements_type(localization, typ))
                return dvar['out']

        return arguments[0]

    @staticmethod
    def zeros(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'order'], {
            'dtype': [type, IterableDataStructure],
            'order': Str,
        }, 'zeros')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar.keys():
            dtype = dvar['dtype']
        else:
            dtype = None

        t = call_utilities.check_possible_values(dvar, 'order', ['C', 'F'])
        if isinstance(t, StypyTypeError):
            return t

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(numpy.float64(), dtype=dtype)
        else:
            if call_utilities.is_iterable(dtype):
                tup = call_utilities.wrap_contained_type(tuple())
                tup.set_contained_type(numpy.float64())
                contents = call_utilities.create_numpy_array(tup)
                return call_utilities.create_numpy_array(contents)

            dims = call_utilities.get_dimensions(localization, arguments[0])
            typ = call_utilities.create_numpy_array_n_dimensions(numpy.float64(), dims, dtype=dtype)

            return typ


    @staticmethod
    def unpackbits(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis'], {
            'axis': Integer,
        }, 'unpackbits')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'axis' in dvar.keys():
            contained = call_utilities.get_contained_elements_type(localization, arguments[0])
        else:
            contained = call_utilities.get_inner_type(localization, arguments[0])

        return call_utilities.create_numpy_array(contained)

