#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, wrap_contained_type
from stypy.types.union_type import UnionType


class TypeModifiers:
    @staticmethod
    def atleast_2d(localization, proxy_obj, arguments):
        rets = list()
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                if call_utilities.get_dimensions(localization, arg) >= 2:
                    rets.append(arg)
                else:
                    rets.append(
                        call_utilities.create_numpy_array_n_dimensions(call_utilities.get_inner_type(localization, arg),
                                                                       2))
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the atleast_2d function".format(
                                          str(arg)))

        if len(rets) == 1:
            return rets[0]

        return tuple(rets)

    @staticmethod
    def hstack(localization, proxy_obj, arguments):
        union = None
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                union = UnionType.add(union, call_utilities.get_inner_type(localization, arg))
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the hstack function".format(str(arg)))

        return call_utilities.create_numpy_array(union)

    @staticmethod
    def vstack(localization, proxy_obj, arguments):
        union = None
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                union = UnionType.add(union, get_contained_elements_type(localization, arg))
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the vstack function".format(str(arg)))

        return call_utilities.create_numpy_array(union)

    @staticmethod
    def dstack(localization, proxy_obj, arguments):
        elem_list = wrap_contained_type(list())
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                cont = get_contained_elements_type(localization, arg)
                if call_utilities.is_iterable(cont):
                    union2 = UnionType.add(elem_list.get_contained_type(), cont.get_contained_type())
                    elem_list.set_contained_type(union2)
                else:
                    union2 = UnionType.add(elem_list.get_contained_type(), cont)
                    elem_list.set_contained_type(union2)
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the dstack function".format(str(arg)))

        return call_utilities.create_numpy_array(elem_list)

    @staticmethod
    def block(localization, proxy_obj, arguments):
        union = None
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                union = UnionType.add(union, get_contained_elements_type(localization, arg))
            else:
                union = UnionType.add(union, arg)

        return call_utilities.create_numpy_array(union)

    @staticmethod
    def concatenate(localization, proxy_obj, arguments):
        union = None
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                union = UnionType.add(union, get_contained_elements_type(localization, arg))
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the concatenate function".format(
                                          str(arg)))

        return call_utilities.create_numpy_array(union)

    @staticmethod
    def stack(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['axis'], {
                                                           'axis': Integer,
                                                       }, 'stack')

        if isinstance(dvar, StypyTypeError):
            return dvar

        union = None
        for arg in arguments:
            if call_utilities.is_iterable(arg):
                union = UnionType.add(union, get_contained_elements_type(localization, arg))
            else:
                return StypyTypeError(localization,
                                      "A non-iterable parameter {0} was passed to the stack function".format(str(arg)))

        return call_utilities.create_numpy_array(union)

    @staticmethod
    def hsplit(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def vsplit(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def dsplit(localization, proxy_obj, arguments):
        l = wrap_contained_type(list())
        l.set_contained_type(call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0])))
        return l

    @staticmethod
    def split(localization, proxy_obj, arguments):
        l = wrap_contained_type(list())
        l.set_contained_type(arguments[0])
        return l

    @staticmethod
    def array_split(localization, proxy_obj, arguments):
        l = wrap_contained_type(list())
        l.set_contained_type(arguments[0])
        return l
