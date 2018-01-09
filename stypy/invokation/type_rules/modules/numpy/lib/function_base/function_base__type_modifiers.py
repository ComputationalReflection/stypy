#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number, IterableDataStructureWithTypedElements
from stypy.invokation.type_rules.type_groups.type_group_generator import RealNumber, Integer
from stypy.types.type_containers import get_contained_elements_type
from stypy.types.type_wrapper import TypeWrapper

class TypeModifiers:
    @staticmethod
    def unwrap(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0], False)
        else:
            return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

    @staticmethod
    def interp(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['xp', 'fp','left', 'right', 'period'],{
            'xp': IterableDataStructureWithTypedElements(RealNumber),
            'fp': IterableDataStructureWithTypedElements(Number),
            'left': Number,
            'right': Number,
            'period': RealNumber,
        }, 'interp')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            return arguments[0]

        return call_utilities.create_numpy_array(numpy.float64())

    @staticmethod
    def i0(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]):
            return arguments[0]
        else:
            if Number == type(arguments[0]):
                return call_utilities.create_numpy_array(arguments[0])
            else:
                return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

    @staticmethod
    def sinc(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return numpy.float64()
        if Number == type(get_contained_elements_type(arguments[0])):
            return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

        return arguments[0]

    @staticmethod
    def gradient(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['vargargs', 'edge_order'],{
            'varargs': IterableDataStructureWithTypedElements(RealNumber),
            'edge_order': Integer,
        }, 'gradient')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

    @staticmethod
    def diff(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['n', 'axis'],{
            'n': Integer,
            'axis': Integer,
        }, 'diff')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

    @staticmethod
    def trapz(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['x', 'dx','axis'],{
            'x': IterableDataStructureWithTypedElements(RealNumber),
            'dx': RealNumber,
            'axis': Integer,
        }, 'trapz')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return numpy.float64()

    @staticmethod
    def angle(localization, proxy_obj, arguments):
        if not call_utilities.is_iterable(arguments):
            return numpy.float64()

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        dims = call_utilities.get_dimensions(localization, r)
        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(numpy.float64(), dims-1)
        else:
            return call_utilities.create_numpy_array(numpy.float64())