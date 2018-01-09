#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.invokation.type_rules.type_groups.type_group_generator import RealNumber, Integer, Str, DynamicType, IterableDataStructureWithTypedElements
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, get_new_type_instance, set_contained_elements_type
from stypy.types import type_containers
from stypy.types.standard_wrapper import wrap_contained_type


class TypeModifiers:
    @staticmethod
    def geterr(localization, proxy_obj, arguments):
        w = wrap_contained_type(dict())
        type_containers.set_contained_elements_type_for_key(w, str(), str())
        return w

    @staticmethod
    def seterr(localization, proxy_obj, arguments):
        pass

    @staticmethod
    def allclose(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['rtol', 'atol','equal_nan'],{
            'rtol': RealNumber,
            'atol': RealNumber,
            'equal_nan': bool,
        }, 'allclose', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        return bool()

    @staticmethod
    def convolve(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['mode'],{
            'mode': Str,
        }, 'convolve', 2)

        dvar = call_utilities.check_possible_values(dvar, 'mode', ['full', 'same', 'valid'])
        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            type1 = arguments[0]
        else:
            type1 = get_contained_elements_type(localization, arguments[0])

        if Number == type(arguments[1]):
            type2 = arguments[1]
        else:
            type2 = get_contained_elements_type(localization, arguments[1])

        return call_utilities.create_numpy_array(call_utilities.cast_to_greater_numpy_type(type1, type2))

    @staticmethod
    def cross(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axisa', 'axisb','axisc', 'axis'],{
            'axisa': Integer,
            'axisb': Integer,
            'axisc': Integer,
            'axis': Integer,
        }, 'cross', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            type1 = arguments[0]
        else:
            type1 = get_contained_elements_type(localization, arguments[0])

        if Number == type(arguments[1]):
            type2 = arguments[1]
        else:
            type2 = get_contained_elements_type(localization, arguments[1])

        return call_utilities.create_numpy_array(call_utilities.cast_to_greater_numpy_type(type1, type2))

    @staticmethod
    def ones(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'order'],{
            'dtype': type,
            'order': Str,
        }, 'ones')

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
            dims = call_utilities.get_dimensions(localization, arguments[0])
            typ = call_utilities.create_numpy_array(numpy.float64(), dtype=dtype)

            for i in range(dims):
                typ = call_utilities.create_numpy_array(typ)

            return typ

    @staticmethod
    def ascontiguousarray(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype'],{
            'dtype': type,
        }, 'ascontiguousarray')

        if 'dtype' in dvar.keys():
            dtype = dvar['dtype']
        else:
            dtype = None

        if call_utilities.is_iterable(arguments[0]):
            typ = get_contained_elements_type(localization, arguments[0])
        else:
            typ = arguments[0]

        return call_utilities.create_numpy_array(typ, dtype=dtype)


    @staticmethod
    def outer(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out'],{
            'out': numpy.ndarray,
        }, 'outer', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        t1 = get_contained_elements_type(localization, arguments[0])
        t2 = get_contained_elements_type(localization, arguments[1])
        l = wrap_contained_type(list())
        set_contained_elements_type(localization, l, call_utilities.cast_to_greater_numpy_type(t1, t2))

        if 'out' in dvar:
            set_contained_elements_type(localization, dvar['out'], l)

        return call_utilities.create_numpy_array(l)

    @staticmethod
    def tensordot(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axes'],{
            'axes': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'tensordot', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        l = wrap_contained_type(list())
        set_contained_elements_type(localization, l, DynamicType())

        return call_utilities.create_numpy_array(l)

    @staticmethod
    def zeros_like(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'order', 'subok'], {
            'dtype': type,
            'order': Str,
            'subok': bool,
        }, 'ones')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar.keys():
            dtype = dvar['dtype']
        else:
            dtype = None

        t = call_utilities.check_possible_values(dvar, 'order', ['C', 'F', 'A', 'K'])
        if isinstance(t, StypyTypeError):
            return t

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0], dtype=dtype)
        else:
            dims = call_utilities.get_dimensions(localization, arguments[0])
            typ = call_utilities.create_numpy_array_n_dimensions(call_utilities.get_inner_type(localization, arguments[0]),
                                                                 dims, dtype=dtype)
            return typ

    @staticmethod
    def roll(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['shift', 'axis'], {
            'shift': [Integer, IterableDataStructureWithTypedElements(Integer)],
            'axis': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'roll', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        if not Number == type(t):
            return StypyTypeError(localization, "The contents of the passed array are not numeric")

        return arguments[0]

    @staticmethod
    def rollaxis(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['shift', 'axis'], {
            'shift': [Integer, IterableDataStructureWithTypedElements(Integer)],
            'axis': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'rollaxis', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        if not Number == type(t):
            return StypyTypeError(localization, "The contents of the passed array are not numeric")

        return arguments[0]