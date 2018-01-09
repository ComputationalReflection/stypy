#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, \
    IterableDataStructure, Str, Number
from stypy.type_inference_programs.stypy_interface import set_contained_elements_type, get_contained_elements_type


class TypeModifiers:
    @staticmethod
    def MaskedArray(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['mask'], {
            'mask': IterableDataStructure,
        }, 'MaskedArray')

        if isinstance(dvar, StypyTypeError):
            return dvar
        # import numpy

        # x = numpy.ndarray([1])
        # mx = numpy.ma.masked_array(x, mask=[0])
        # return mx
        # # t = call_utilities.get_inner_type(localization, arguments[0])
        # # t = 0.0
        # #
        # # return numpy.ma.masked_array(numpy.ndarray(t), [True], t)

        # arr1 = call_utilities.create_numpy_array_n_dimensions(
        #     call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, arguments[0])),
        # call_utilities.get_dimensions(localization, arguments[0]),  call_utilities.get_inner_type(localization, arguments[0]), False)
        #
        # if 'mask' in dvar.keys():
        #     arr2 = call_utilities.create_numpy_array_n_dimensions(
        #         call_utilities.cast_to_numpy_type(call_utilities.get_inner_type(localization, dvar['mask'])),
        #         call_utilities.get_dimensions(localization, dvar['mask']),  call_utilities.get_inner_type(localization, arguments[0]), False)
        #
        #     arr = numpy.ma.masked_array(arr1, arr2)
        # else:
        #     arr = numpy.ma.masked_array(arr1)
        #
        # return arr
        mask = call_utilities.wrap_contained_type(numpy.ma.core.MaskedArray(arguments[0], mask=[False]))
        mask.set_contained_type(mask.wrapped_type.min())
        return mask

    @staticmethod
    def log(localization, proxy_obj, arguments):
        mask = call_utilities.wrap_contained_type(numpy.ma.core.MaskedArray(arguments[0], mask=[False]))
        mask.set_contained_type(mask.wrapped_type.min())
        return mask