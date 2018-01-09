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
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type, get_sample_instance_for_type, wrap_contained_type
from stypy.types.union_type import UnionType

class TypeModifiers:
    @staticmethod
    def linspace(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['num',
            'endpoint'
            'retstep'
            'dtype'], {
            'num': Integer,
            'endpoint': bool,
            'retstep': bool,
            'dtype': type,
        }, 'linspace', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        temp_ret = call_utilities.create_numpy_array(numpy.float64())
        if 'retstep' in dvar.keys():
            union = UnionType.add(temp_ret, numpy.float64())
            out_tuple = wrap_contained_type((union,))
            out_tuple.set_contained_type(union)
            return out_tuple

        return temp_ret