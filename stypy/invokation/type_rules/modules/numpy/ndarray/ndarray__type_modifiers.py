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
    @staticmethod
    def __getitem__(localization, proxy_obj, arguments):
        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        index_selector = arguments[0]
        if isinstance(index_selector, tuple):
            index_selector = index_selector[0]

        dims = call_utilities.get_dimensions(localization, r)
        if dims > 1:
            if call_utilities.is_iterable(arguments[0]):
                if call_utilities.is_iterable(arguments[0].get_contained_type()):
                    return call_utilities.create_numpy_array_n_dimensions(
                        call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, r)), dims - 1)

            contained = call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, r))
            for i in range(dims - 1):
                contained = UnionType.add(contained,
                                          call_utilities.create_numpy_array_n_dimensions(r.get_contained_type(), i + 1))
        else:
            contained = r.get_contained_type()

        if isinstance(index_selector, TypeWrapper):
            if isinstance(index_selector.wrapped_type, slice) or (
                        call_utilities.is_iterable(index_selector) and not isinstance(index_selector.wrapped_type,
                                                                                      tuple)):
                l = call_utilities.create_numpy_array(contained)
                return l
        return contained  # proxy_obj.__self__.dtype.type()

    @staticmethod
    def __setitem__(localization, proxy_obj, arguments):
        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        existing = r.get_contained_type()
        if existing is types.NoneType or type(existing) is types.NoneType:
            r.set_contained_type(arguments[1])
        else:
            r.set_contained_type(UnionType.add(arguments[1], existing))

    @staticmethod
    def argmin(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return numpy.int32()

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'out'], {
            'axis': Integer,
            'out': numpy.ndarray,
        }, 'argmin', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], numpy.int32())
            return dvar['out']

        return call_utilities.create_numpy_array(numpy.int32())

    @staticmethod
    def view(localization, proxy_obj, arguments):
        if len(arguments) == 0:
            return arguments[0]

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['dtype', 'type'], {
            'dtype': [type, IterableDataStructure],
            'type': type,
        }, 'view', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        try:
            contained_type = proxy_obj.__self__[0]

            if 'dtype' in dvar.keys():
                if not issubclass(arguments[0], numpy.ndarray):
                    return call_utilities.create_numpy_array(contained_type, dvar['dtype'])
                else:
                    return call_utilities.create_numpy_array(contained_type).view(dvar['dtype'])

            return call_utilities.create_numpy_array(contained_type)
        except:
            return wrap_type(proxy_obj.__self__)

    @staticmethod
    def reshape(localization, proxy_obj, arguments):
        shape_levels = len(arguments)
        if Str == type(arguments[-1]):
            shape_levels -= 1

        if shape_levels == 0:
            return StypyTypeError(localization,
                                  "Invalid 'shape' parameter for reshape call: There must be at least one shape element")
        if call_utilities.is_iterable(arguments[0]):
            shape_levels = 2

        if len(arguments) > 1 and not Str == type(arguments[-1]) and not Integer == type(arguments[-1]):
            return StypyTypeError(localization,
                                  "Invalid 'order' parameter for reshape call: {0}".format(str(arguments[-1])))

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        contained = r.get_contained_type()

        for i in range(shape_levels):
            contained = call_utilities.create_numpy_array(contained)
        return contained

    @staticmethod
    def dot(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['out'], {
                                                           'out': numpy.ndarray,
                                                       }, 'dot')

        if isinstance(dvar, StypyTypeError):
            return dvar

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)

        if Number == type(arguments[0]):
            c_t = get_contained_elements_type(localization, r)
            if not 'out' in dvar.keys():
                return call_utilities.create_numpy_array(c_t)
            else:
                set_contained_elements_type(localization, dvar['out'], c_t)
                return dvar['out']

        if call_utilities.is_iterable(arguments[0]):
            if call_utilities.get_dimensions(localization, r) == 1 and call_utilities.get_dimensions(
                    localization, arguments[0]) == 1:
                return call_utilities.cast_to_greater_numpy_type(
                    call_utilities.get_inner_type(localization, r),
                    call_utilities.get_inner_type(localization, arguments[0]))

            typ = call_utilities.cast_to_greater_numpy_type(call_utilities.get_inner_type(localization, r),
                                                            call_utilities.get_inner_type(localization, arguments[0]))
            for i in range(call_utilities.get_dimensions(localization, r)):
                typ = call_utilities.create_numpy_array(typ)

            if not 'out' in dvar.keys():
                return typ
            else:
                set_contained_elements_type(localization, dvar['out'], get_contained_elements_type(localization, typ))
                return dvar['out']

        return r

    @staticmethod
    def mean(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out', 'keepdims'], {
            'dtype': type,
            'axis': Integer,
            'out': numpy.ndarray,
            'keepdims': bool,
        }, 'mean', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        dims = call_utilities.get_dimensions(localization, r)

        if 'axis' in dvar:
            if dims > 1:
                if 'out' in dvar.keys():
                    set_contained_elements_type(localization, dvar['out'], numpy.int32())
                    return dvar['out']
                return call_utilities.create_numpy_array(numpy.float64())

        return numpy.float64()

    @staticmethod
    def astype(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['dtype', 'order', 'casting', 'subok', 'copy'], {
                                                           'dtype': type,
                                                           'order': Str,
                                                           'casting': Str,
                                                           'subok': bool,
                                                           'copy': bool,
                                                       }, 'astype', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        dvar = call_utilities.check_possible_values(dvar, 'order', ['C', 'F', 'A', 'K'])
        if isinstance(dvar, StypyTypeError):
            return dvar

        dvar = call_utilities.check_possible_values(dvar, 'casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
        if isinstance(dvar, StypyTypeError):
            return dvar

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        dims = call_utilities.get_dimensions(localization, r)

        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(
                call_utilities.get_contained_elements_type(localization, r),
                dims - 1, dvar['dtype'])
        else:
            return call_utilities.create_numpy_array(call_utilities.get_contained_elements_type(localization, r),
                                                     dvar['dtype'])

    @staticmethod
    def repeat(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['repeats', 'axis'], {
                                                           'repeats': [Integer,
                                                                       IterableDataStructureWithTypedElements(Integer)],
                                                           'axis': Integer,
                                                       }, 'repeat')

        if isinstance(dvar, StypyTypeError):
            return dvar

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        if 'axis' in dvar.keys():
            return r
        else:
            typ = call_utilities.get_inner_type(localization, r)
            return call_utilities.create_numpy_array(typ)

    @staticmethod
    def sum(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out'], {
            'axis': [types.NoneType, Integer, IterableDataStructureWithTypedElements(Integer)],
            'dtype': type,
            'out': numpy.ndarray,
            'keepdims': bool}, 'sum')

        if isinstance(dvar, StypyTypeError):
            return dvar

        r = TypeWrapper.get_wrapper_of(proxy_obj.__self__)
        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'],
                                        get_contained_elements_type(localization, r))
            return dvar['out']

        if 'axis' in dvar.keys():
            return call_utilities.create_numpy_array(get_contained_elements_type(localization, r))

        return call_utilities.cast_to_numpy_type(get_contained_elements_type(localization, r))

    @staticmethod
    def nonzero(localization, proxy_obj, arguments):
        arr = call_utilities.create_numpy_array(numpy.int64())
        t = call_utilities.wrap_contained_type(tuple())
        t.set_contained_type(arr)

        return t

    @staticmethod
    def __div__(localization, proxy_obj, arguments, func_name='__div__'):
        # dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['out', 'where'], {
        #     'out': numpy.ndarray,
        #     'where': numpy.ndarray,
        # }, func_name, 2)
        #
        # if isinstance(dvar, StypyTypeError):
        #     return dvar
        try:
            dims = 0
            if call_utilities.is_iterable(arguments[0]):
                param0 = call_utilities.get_inner_type(localization, arguments[0])
                dims = call_utilities.get_dimensions(localization, arguments[0])
            else:
                param0 = arguments[0]

            if dims > 0:
                ret = call_utilities.create_numpy_array_n_dimensions(
                    call_utilities.cast_to_numpy_type(param0),
                    dims)
            else:
                ret = call_utilities.create_numpy_array(call_utilities.cast_to_numpy_type(param0))

        except Exception as ex:
            return StypyTypeError(localization, str(ex))

        # if 'out' in dvar.keys():
        #     set_contained_elements_type(localization, dvar['out'],
        #                                 ret)
        #     return dvar['out']

        return ret

    @staticmethod
    def __mod__(localization, proxy_obj, arguments):
        return TypeModifiers.__div__(localization, proxy_obj, arguments, func_name='__mod__')

    @staticmethod
    def __floordiv__(localization, proxy_obj, arguments):
        return TypeModifiers.__div__(localization, proxy_obj, arguments, func_name='__floordiv__')

    @staticmethod
    def __rdiv__(localization, proxy_obj, arguments):
        return TypeModifiers.__div__(localization, proxy_obj, arguments, func_name='__rdiv__')