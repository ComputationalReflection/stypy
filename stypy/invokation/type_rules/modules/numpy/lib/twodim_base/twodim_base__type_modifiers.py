#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer


class TypeModifiers:
    @staticmethod
    def diag(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['k'], {
            'k': Integer,
        }, 'diag')

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        dims = call_utilities.get_dimensions(localization, arguments[0])

        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(t, dims - 1)

        return call_utilities.create_numpy_array(t)

    @staticmethod
    def triu(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['k'], {
            'k': Integer,
        }, 'triu')

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        dims = call_utilities.get_dimensions(localization, arguments[0])

        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(t, dims - 1)

        return call_utilities.create_numpy_array(t)

    @staticmethod
    def tril(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['k'], {
            'k': Integer,
        }, 'tril')

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        dims = call_utilities.get_dimensions(localization, arguments[0])

        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(t, dims - 1)

        return call_utilities.create_numpy_array(t)

    @staticmethod
    def diagflat(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['k'], {
            'k': Integer,
        }, 'diagflat')

        if isinstance(dvar, StypyTypeError):
            return dvar

        t = call_utilities.get_inner_type(localization, arguments[0])
        dims = call_utilities.get_dimensions(localization, arguments[0])

        if dims > 1:
            return call_utilities.create_numpy_array_n_dimensions(t, dims - 1)

        return call_utilities.create_numpy_array(t)