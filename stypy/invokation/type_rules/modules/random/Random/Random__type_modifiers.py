

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, RealNumber, \
    IterableDataStructureWithTypedElements, IterableDataStructure


class TypeModifiers:
    @staticmethod
    def uniform(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return float()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['low', 'high', 'size'],{
            'low': [RealNumber, IterableDataStructureWithTypedElements(RealNumber)],
            'high': [RealNumber, IterableDataStructureWithTypedElements(RealNumber)],
            'size': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'uniform', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if call_utilities.check_parameter_type(dvar, 'low', IterableDataStructureWithTypedElements(RealNumber)) or \
                call_utilities.check_parameter_type(dvar, 'high', IterableDataStructureWithTypedElements(RealNumber)) or \
                call_utilities.check_parameter_type(dvar, 'size', IterableDataStructureWithTypedElements(RealNumber)):
            return call_utilities.create_numpy_array(numpy.float64())

        if 'size' in dvar.keys():
            return call_utilities.create_numpy_array(numpy.float64())

        return float()

    @staticmethod
    def random(localization, proxy_obj, arguments):
        # if len(arguments) == 0:
        #     return float()
        #
        # dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['size'], {
        #     'size': [Integer, IterableDataStructureWithTypedElements(Integer)],
        # }, 'random', 0)
        #
        # if isinstance(dvar, StypyTypeError):
        #     return dvar
        #
        # if call_utilities.is_iterable(arguments[0]):
        #     inner_array = call_utilities.create_numpy_array(numpy.float64())
        #     return call_utilities.create_numpy_array(inner_array)
        #
        # return float()
        if len(arguments) == 0:
            return float()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['size'], {
            'size': [Integer, IterableDataStructureWithTypedElements(Integer)],
        }, 'random', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if call_utilities.is_iterable(arguments[0]):
            inner_array = call_utilities.create_numpy_array_n_dimensions(numpy.float64(), call_utilities.get_dimensions(localization, arguments[0]))
            return call_utilities.create_numpy_array(inner_array)

        return call_utilities.create_numpy_array(numpy.float64())

    @staticmethod
    def randint(localization, proxy_obj, arguments):
        if len(arguments) == 1:
            return int()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['high', 'size', 'dtype'], {
            'high': Integer,
            'size': [Integer, IterableDataStructureWithTypedElements(Integer)],
            'dtype': type,
        }, 'randint', 1)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'dtype' in dvar:
            if not Integer == dvar['dtype']:
                return StypyTypeError(localization, "Unsupported type {0} for randint".format(str(dvar['dtype'])))
        if 'size' in dvar:
            return call_utilities.create_numpy_array(numpy.int32())

        return int()

    @staticmethod
    def choice(localization, proxy_obj, arguments):
        return call_utilities.get_contained_elements_type(localization, arguments[0])
