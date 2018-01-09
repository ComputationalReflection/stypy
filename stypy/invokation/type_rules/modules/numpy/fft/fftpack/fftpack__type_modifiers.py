#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, Number, Str, \
    IterableDataStructureWithTypedElements
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type

class TypeModifiers:
    @staticmethod
    def fft(localization, proxy_obj, arguments, f_name='fft', use_dims=False):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['n', 'axis', 'norm'], {
            'n': Integer,
            'axis': Integer,
            'norm': Str
        }, f_name)

        if isinstance(dvar, StypyTypeError):
            return dvar

        temp = call_utilities.check_possible_values(dvar, 'norm', 'ortho')
        if isinstance(temp, StypyTypeError):
            return temp

        if use_dims:
            dims = call_utilities.get_dimensions(localization, arguments[0])
        else:
            dims = 1

        typ = numpy.complex128()
        for i in range(dims):
            typ = call_utilities.create_numpy_array(typ)

        return typ

    @staticmethod
    def ifft(localization, proxy_obj, arguments):
        return TypeModifiers.fft(localization, proxy_obj, arguments, 'ifft')

    @staticmethod
    def rfft(localization, proxy_obj, arguments):
        return TypeModifiers.fft(localization, proxy_obj, arguments, 'rfft')

    @staticmethod
    def irfft(localization, proxy_obj, arguments):
        return TypeModifiers.fft(localization, proxy_obj, arguments, 'irfft')

    @staticmethod
    def fftn(localization, proxy_obj, arguments, f_name='fftn', use_dims=False):
        if Number == type(arguments[0]):
            return call_utilities.create_numpy_array(arguments[0])

        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['n', 'axis', 'norm'], {
            'n': IterableDataStructureWithTypedElements(Integer),
            'axis': IterableDataStructureWithTypedElements(Integer),
            'norm': Str
        }, f_name)

        if isinstance(dvar, StypyTypeError):
            return dvar

        temp = call_utilities.check_possible_values(dvar, 'norm', 'ortho')
        if isinstance(temp, StypyTypeError):
            return temp

        if use_dims:
            dims = call_utilities.get_dimensions(localization, arguments[0])
        else:
            dims = 1

        typ = numpy.complex128()
        for i in range(dims):
            typ = call_utilities.create_numpy_array(typ)

        return typ

    @staticmethod
    def fft2(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'fft2', True)

    @staticmethod
    def ifft2(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'ifft2', True)

    @staticmethod
    def ifftn(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'ifftn', True)

    @staticmethod
    def rfft2(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'rfft2', True)

    @staticmethod
    def irfft2(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'irfft2', True)

    @staticmethod
    def rfftn(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'rfftn', True)

    @staticmethod
    def irfftn(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'irfftn', True)

    @staticmethod
    def ihfft(localization, proxy_obj, arguments):
        return TypeModifiers.fftn(localization, proxy_obj, arguments, 'fft')

    @staticmethod
    def hfft(localization, proxy_obj, arguments, f_name='hfft', use_dims=False):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['n', 'axis', 'norm'], {
            'n': Integer,
            'axis': Integer,
            'norm': Str
        }, f_name)

        if isinstance(dvar, StypyTypeError):
            return dvar

        temp = call_utilities.check_possible_values(dvar, 'norm', 'ortho')
        if isinstance(temp, StypyTypeError):
            return temp

        if use_dims:
            dims = call_utilities.get_dimensions(localization, arguments[0])
        else:
            dims = 1

        typ = numpy.float64()
        for i in range(dims):
            typ = call_utilities.create_numpy_array(typ)

        return typ

    @staticmethod
    def fftfreq(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(numpy.float64)

    @staticmethod
    def rfftfreq(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(numpy.float64)

    @staticmethod
    def fftshift(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))

    @staticmethod
    def rfftshift(localization, proxy_obj, arguments):
        return call_utilities.create_numpy_array(get_contained_elements_type(localization, arguments[0]))