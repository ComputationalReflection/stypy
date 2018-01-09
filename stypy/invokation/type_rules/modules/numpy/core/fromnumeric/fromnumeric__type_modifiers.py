#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy
from numpy.core.numeric import asarray

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, Str, \
    IterableDataStructureWithTypedElements
from stypy.invokation.type_rules.type_groups.type_group_generator import Number, IterableDataStructure
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type


class TypeModifiers:
    @staticmethod
    def __instance_NoneTypes(arg):
        if arg is types.NoneType:
            return None
        else:
            return arg

    @staticmethod
    def _wrapit(localization, proxy_obj, arguments):
        arguments = map(lambda arg: TypeModifiers.__instance_NoneTypes(arg), arguments)
        try:
            if not hasattr(arguments[0], '__array_wrap__'):
                if hasattr(asarray(arguments[0]), arguments[1]):
                    to_invoke = getattr(asarray(arguments[0]), arguments[1])
                    if isinstance(arguments[-1], dict):
                        kwds = arguments[1]
                        args = arguments[2:-1]
                    else:
                        kwds = dict()
                        args = arguments[2:]

                    return to_invoke(*args, **kwds)
            return proxy_obj(localization, *arguments)
        except:
            return proxy_obj(localization, *arguments)

    @staticmethod
    def around(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['decimals', 'out'], {
            'decimals': Integer,
            'out': numpy.ndarray,
        }, 'around')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def round_(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['decimals', 'out'], {
            'decimals': Integer,
            'out': numpy.ndarray,
        }, 'round_')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def clip(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['a_min', 'a_max', 'out'], {
            'a_min': [Integer, IterableDataStructureWithTypedElements(Integer), types.NoneType],
            'a_max': [Integer, IterableDataStructureWithTypedElements(Integer), types.NoneType],
            'out': numpy.ndarray,
        }, 'clip')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def reshape(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0], False)

        shape_levels = len(arguments) - 1
        if Str == type(arguments[-1]):
            shape_levels -= 1

        if shape_levels == 0:
            return StypyTypeError(localization,
                                  "Invalid 'shape' parameter for reshape call: There must be at least one shape element")

        if IterableDataStructure == type(arguments[1]):
            shape_levels = 2

        if len(arguments) > 2 and not Str == type(arguments[-1]) and not Integer == type(arguments[-1]):
            return StypyTypeError(localization,
                                  "Invalid 'order' parameter for reshape call: {0}".format(str(arguments[-1])))

        contained = get_contained_elements_type(localization, arguments[0])

        for i in range(shape_levels):
            contained = call_utilities.create_numpy_array(contained)
        return contained

        # dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['newshape', 'order'],{
        #     'newshape': [Integer, IterableDataStructureWithTypedElements(Integer)],
        #     'order': Str,
        # }, 'reshape')
        #
        # if isinstance(dvar, StypyTypeError):
        #     return dvar
        #
        # return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def sum(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out'], {
            'axis': [types.NoneType, Integer, IterableDataStructureWithTypedElements(Integer)],
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'sum')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

        return call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, arguments[0]))

    @staticmethod
    def prod(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out'], {
            'axis': [types.NoneType, Integer, IterableDataStructureWithTypedElements(Integer)],
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'prod')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

        return call_utilities.cast_to_numpy_type(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def cumsum(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(call_utilities.cast_to_numpy_type(arguments[0]))

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'out'], {
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'cumsum')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def cumprod(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(call_utilities.cast_to_numpy_type(arguments[0]))

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'out'], {
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'cumprod')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def argmin(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return numpy.int32()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'out'], {
            'axis': Integer,
            'out': numpy.ndarray,
        }, 'argmin')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], numpy.int32())
            return dvar['out']

        return call_utilities.create_numpy_array(numpy.int32())

    @staticmethod
    def all(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return numpy.bool_()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'out', 'keepdims'], {
            'axis': [types.NoneType, Integer, IterableDataStructureWithTypedElements(Integer)],
            'out': numpy.ndarray,
            'keepdims': bool}, 'all')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], numpy.bool_())
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(numpy.bool_())
        return numpy.bool_()

    @staticmethod
    def any(localization, proxy_obj, arguments):
        if Number == type(arguments[0]):
            return numpy.bool_()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'out', 'keepdims'], {
            'axis': [types.NoneType, Integer, IterableDataStructureWithTypedElements(Integer)],
            'out': numpy.ndarray,
            'keepdims': bool}, 'any')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], numpy.bool_())
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(numpy.bool_())
        return numpy.bool_()

    @staticmethod
    def argpartition(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['kth', 'axis', 'kind', 'order'], {
            'kth': [Integer, IterableDataStructureWithTypedElements(Integer)],
            'axis': Integer,
            'kind': Str,
            'order': [Str, IterableDataStructureWithTypedElements(Str)]}, 'argpartition')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0])

        if call_utilities.is_numpy_array(arguments[0]):
            return arguments[0]
        return call_utilities.create_numpy_array(arguments[0])

    @staticmethod
    def argsort(localization, proxy_obj, arguments, fname='argsort'):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'kind', 'order'], {
            'axis': Integer,
            'kind': Str,
            'order': [Str, IterableDataStructureWithTypedElements(Str)]}, fname)

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.check_possible_values(dvar, 'kind', ['quicksort', 'mergesort', 'heapsort'])
        if isinstance(t, StypyTypeError):
            return t

        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0])

        if call_utilities.is_numpy_array(arguments[0]):
            return arguments[0]
        return call_utilities.create_numpy_array(arguments[0])

    @staticmethod
    def trace(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['offset', 'axis1', ' axis2', 'dtype', 'out'], {
                                                           'offset': Integer,
                                                           'axis1': Integer,
                                                           'axis2': Integer,
                                                           'dtype': type,
                                                           'out': numpy.ndarray,
                                                       }, 'trace')

        if isinstance(dvar, StypyTypeError):
            return dvar

        dim = call_utilities.get_dimensions(localization, arguments[0])
        if dim == 1:
            return call_utilities.cast_to_numpy_type(get_contained_elements_type(localization, arguments[0]))
        else:
            ret = call_utilities.create_numpy_array(call_utilities.get_inner_type(localization, arguments[0]))

        if 'out' in dvar.keys():
            if dim == 1 or not (call_utilities.get_dimensions(localization, dvar['out']) == 1):
                return StypyTypeError(localization, "Wrong dimensions of out parameter in trace call")

            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return ret

    @staticmethod
    def repeat(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['repeats', 'axis'], {
                                                           'repeats': [Integer, IterableDataStructureWithTypedElements(Integer)],
                                                           'axis': Integer,
                                                       }, 'repeat')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'axis' in dvar.keys():
            return arguments[0]
        else:
            typ = call_utilities.get_inner_type(localization, arguments[0])
            return call_utilities.create_numpy_array(typ)

    @staticmethod
    def sort(localization, proxy_obj, arguments):
        return TypeModifiers.argsort(localization, proxy_obj, arguments, fname='sort')

    @staticmethod
    def nonzero(localization, proxy_obj, arguments):
        arr = call_utilities.create_numpy_array(numpy.int64())
        t = call_utilities.wrap_contained_type(tuple())
        t.set_contained_type(arr)

        return t

    @staticmethod
    def flatnonzero(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(numpy.int64())

    @staticmethod
    def count_nonzero(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['axis'], {
                                                           'axis': Integer,
                                                       }, 'count_nonzero')

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(numpy.int64())
        else:
            return numpy.int64()