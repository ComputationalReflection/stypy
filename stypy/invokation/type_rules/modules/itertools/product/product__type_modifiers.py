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
import itertools


class TypeModifiers:

    @staticmethod
    def product(localization, proxy_obj, arguments):
        return_tuple = call_utilities.wrap_contained_type(tuple())
        tuple_contents = None

        for arg in arguments:
            if Str == type(arg):
                tuple_contents = UnionType.add(tuple_contents, arg)
            else:
                if type(arg) is not dict:
                    tuple_contents = UnionType.add(tuple_contents, get_contained_elements_type(localization, arg))

        return_tuple.set_contained_type(tuple_contents)
        ret = call_utilities.wrap_contained_type(itertools.product())
        ret.set_contained_type(return_tuple)
        return ret

