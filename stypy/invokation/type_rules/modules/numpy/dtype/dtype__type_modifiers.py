#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.invokation.type_rules.type_groups.type_group_generator import RealNumber, Integer, Str, DynamicType
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, get_new_type_instance
from stypy.types import type_containers
from stypy.types.standard_wrapper import wrap_contained_type
from stypy.types.union_type import UnionType

class TypeModifiers:
    @staticmethod
    def dtype(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['align', 'copy'],{
            'align': bool,
            'copy': bool,
        }, 'dtype')

        if isinstance(dvar, StypyTypeError):
            return dvar

        try:
            if call_utilities.is_iterable(arguments[0]):
                if isinstance(get_contained_elements_type(localization, arguments[0]), UnionType):
                    return numpy.dtype(get_contained_elements_type(localization, arguments[0]).types[0])
                return numpy.dtype(get_contained_elements_type(localization, arguments[0]))
            else:
                return numpy.dtype(arguments[0])
        except:
            return DynamicType()