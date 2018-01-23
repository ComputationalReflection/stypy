#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy
from numpy.core.numeric import asarray

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, Str, \
    IterableDataStructureWithTypedElements
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type, get_sample_instance_for_type
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.modules.numpy import numpy__type_modifiers

class TypeModifiers:
    @staticmethod
    def eq(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]) and call_utilities.is_numpy_array(arguments[1]):
            return call_utilities.create_numpy_array_n_dimensions(bool(), call_utilities.get_dimensions(localization, arguments[0]))
        return None # Type rule results

    @staticmethod
    def __and__(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]) or call_utilities.is_numpy_array(arguments[1]):
            return numpy__type_modifiers.TypeModifiers.bitwise_and(localization, proxy_obj, arguments)

        return None # Type rule results

    @staticmethod
    def __or__(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]) or call_utilities.is_numpy_array(arguments[1]):
            return numpy__type_modifiers.TypeModifiers.bitwise_or(localization, proxy_obj, arguments)

        return None # Type rule results

    @staticmethod
    def __xor__(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]) or call_utilities.is_numpy_array(arguments[1]):
            return numpy__type_modifiers.TypeModifiers.bitwise_xor(localization, proxy_obj, arguments)

        return None # Type rule results

    @staticmethod
    def pow(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[1]):
            return arguments[1]
        if call_utilities.is_numpy_array(arguments[0]):
            return arguments[0]

        return None # Type rule results

    @staticmethod
    def and_(localization, proxy_obj, arguments):
        return TypeModifiers.__and__(localization, proxy_obj, arguments)

    @staticmethod
    def add(localization, proxy_obj, arguments):
        if call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            if isinstance(arguments[0].get_wrapped_type(), tuple) and isinstance(arguments[1].get_wrapped_type(), tuple):
                t1 = call_utilities.get_contained_elements_type(localization, arguments[0])
                t2 = call_utilities.get_contained_elements_type(localization, arguments[1])
                if isinstance(t1, UnionType):
                    t1 = t1.duplicate()
                tEnd = UnionType.add(t1, t2)
                wrap = call_utilities.wrap_contained_type((tEnd,))
                wrap.set_contained_type(tEnd)
                return wrap

        return None # Type rule results

    @staticmethod
    def iadd(localization, proxy_obj, arguments):
        if call_utilities.is_iterable(arguments[0]) and call_utilities.is_iterable(arguments[1]):
            if isinstance(arguments[0].get_wrapped_type(), list) and isinstance(arguments[1].get_wrapped_type(), tuple):
                t1 = get_contained_elements_type(localization, arguments[0])
                t2 = get_contained_elements_type(localization, arguments[1])

                tEnd = UnionType.add(t1, t2)

                set_contained_elements_type(localization, arguments[0], tEnd)
                return arguments[0]

        return None