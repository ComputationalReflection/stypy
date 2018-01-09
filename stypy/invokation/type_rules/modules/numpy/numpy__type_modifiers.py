#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, \
    IterableDataStructureWithTypedElements, Str, Number
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, get_contained_elements_type


class TypeModifiers:
    @staticmethod
    def logical_not(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'], {
            'out': numpy.ndarray,
        }, 'logical_not')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            return bool()

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], bool())
            return dvar['out']

        return call_utilities.create_numpy_array(bool())

    @staticmethod
    def logical_and(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'], {
            'out': numpy.ndarray,
        }, 'logical_and', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return bool()

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], bool())
            return dvar['out']

        return call_utilities.create_numpy_array(bool())

    @staticmethod
    def logical_or(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'], {
            'out': numpy.ndarray,
        }, 'logical_or', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return bool()

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], bool())
            return dvar['out']

        return call_utilities.create_numpy_array(bool())

    @staticmethod
    def logical_xor(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'], {
            'out': numpy.ndarray,
        }, 'logical_xor', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return bool()

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], bool())
            return dvar['out']

        return call_utilities.create_numpy_array(bool())

    @staticmethod
    def negative(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'], {
            'out': numpy.ndarray,
        }, 'logical_not')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def lookfor(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['module', 'import_modules',
                                                                                 'regenerate', 'output'], {
                                                           'module': [Str, IterableDataStructureWithTypedElements(Str)],
                                                           'import_modules': bool,
                                                           'regenerate': bool,
                                                           'output': file,
                                                       }, 'lookfor')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return types.NoneType

    @staticmethod
    def bitwise_and(localization, proxy_obj, arguments, func_name='bitwise_and'):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['where'], {
            'where': [Str, IterableDataStructureWithTypedElements(bool)],
        }, func_name, 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            if not call_utilities.check_possible_types(call_utilities.get_inner_type(localization, arguments[0]),
                                                       [bool, Integer]) or not call_utilities.check_possible_types(
                call_utilities.get_inner_type(localization, arguments[1]), [bool, Integer]):
                return StypyTypeError(localization,
                                      " ufunc '" + func_name + "' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")

            if call_utilities.is_numpy_array(arguments[0]):
                return call_utilities.create_numpy_array(call_utilities.get_inner_type(localization, arguments[0]))
            if call_utilities.is_numpy_array(arguments[1]):
                return call_utilities.create_numpy_array(call_utilities.get_inner_type(localization, arguments[1]))
            return arguments[0]
        else:
            if call_utilities.is_iterable(arguments[0]) and not call_utilities.is_iterable(arguments[1]):
                if not call_utilities.check_possible_types(call_utilities.get_inner_type(localization, arguments[0]),
                                                           [bool, Integer]) or not call_utilities.check_possible_types(
                    arguments[1], [bool, Integer]):
                    return StypyTypeError(localization,
                                          " ufunc '" + func_name + "' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")

                if call_utilities.is_numpy_array(arguments[0]):
                    return call_utilities.create_numpy_array(call_utilities.get_inner_type(localization, arguments[0]))
                return arguments[0]
            else:
                if not call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
                    if not call_utilities.check_possible_types(
                            call_utilities.get_inner_type(localization, arguments[1]),
                            [bool, Integer]) or not call_utilities.check_possible_types(arguments[0], [bool, Integer]):
                        return StypyTypeError(localization,
                                              " ufunc '" + func_name + "' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")

                    if call_utilities.is_numpy_array(arguments[1]):
                        return call_utilities.create_numpy_array(
                            call_utilities.get_inner_type(localization, arguments[1]))
                    return arguments[1]
                else:
                    return arguments[0]

    @staticmethod
    def bitwise_or(localization, proxy_obj, arguments):
        return TypeModifiers.bitwise_and(localization, proxy_obj, arguments, 'bitwise_or')

    @staticmethod
    def bitwise_xor(localization, proxy_obj, arguments):
        return TypeModifiers.bitwise_and(localization, proxy_obj, arguments, 'bitwise_xor')

    @staticmethod
    def reciprocal(localization, proxy_obj, arguments, func_name='reciprocal'):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out', 'where'], {
            'out': numpy.ndarray,
            'where': numpy.ndarray,
        }, func_name)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            ret = call_utilities.cast_to_numpy_type(numpy.float64())
        else:
            try:
                ret = call_utilities.create_numpy_array_n_dimensions(
                    call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, arguments[0])),
                 call_utilities.get_dimensions(localization, arguments[0]))

            except Exception as ex:
                return StypyTypeError(localization, str(ex))

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        ret)
            return dvar['out']

        return ret

    @staticmethod
    def divide(localization, proxy_obj, arguments, func_name='divide'):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out', 'where'], {
            'out': numpy.ndarray,
            'where': numpy.ndarray,
        }, func_name, 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]) and Number == type(arguments[1]):
            return call_utilities.cast_to_greater_numpy_type(arguments[0], arguments[1])

        try:
            dims = 0
            if call_utilities.is_iterable(arguments[0]):
                param0 = call_utilities.get_inner_type(localization, arguments[0])
                dims = call_utilities.get_dimensions(localization, arguments[0])
            else:
                param0 = arguments[0]

            if call_utilities.is_iterable(arguments[1]):
                param1 = call_utilities.get_inner_type(localization, arguments[1])
                temp = call_utilities.get_dimensions(localization, arguments[1])
                if temp > dims:
                    dims = temp
            else:
                param1 = arguments[1]
            if dims > 0:
                ret = call_utilities.create_numpy_array_n_dimensions(
                    call_utilities.cast_to_greater_numpy_type(param0, param1),
                    dims)
            else:
                ret = call_utilities.cast_to_greater_numpy_type(param0, param1)

        except Exception as ex:
            return StypyTypeError(localization, str(ex))

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        ret)
            return dvar['out']

        return ret


    @staticmethod
    def true_divide(localization, proxy_obj, arguments):
        return TypeModifiers.divide(localization, proxy_obj, arguments, func_name='true_divide')

    @staticmethod
    def floor_divide(localization, proxy_obj, arguments):
        return TypeModifiers.divide(localization, proxy_obj, arguments, func_name='floor_divide')

    @staticmethod
    def fmod(localization, proxy_obj, arguments):
        return TypeModifiers.divide(localization, proxy_obj, arguments, func_name='fmod')

    @staticmethod
    def remainder(localization, proxy_obj, arguments):
        return TypeModifiers.divide(localization, proxy_obj, arguments, func_name='remainder')

    @staticmethod
    def log(localization, proxy_obj, arguments):
        return TypeModifiers.reciprocal(localization, proxy_obj, arguments, func_name='log')

    @staticmethod
    def log2(localization, proxy_obj, arguments):
        return TypeModifiers.reciprocal(localization, proxy_obj, arguments, func_name='log2')

    @staticmethod
    def log10(localization, proxy_obj, arguments):
        return TypeModifiers.reciprocal(localization, proxy_obj, arguments, func_name='log10')