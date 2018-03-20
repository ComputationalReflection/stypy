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
from stypy.types.undefined_type import UndefinedType
import itertools


class TypeModifiers:

    @staticmethod
    def heappush(localization, proxy_obj, arguments):
        ex_type = get_contained_elements_type(localization, arguments[0])
        if ex_type is UndefinedType:
            u = arguments[1]
        else:
            u = UnionType.add(ex_type, arguments[1])
        set_contained_elements_type(localization, arguments[0], u)
        return types.NoneType

    @staticmethod
    def heappop(localization, proxy_obj, arguments):
        return get_contained_elements_type(localization, arguments[0])
