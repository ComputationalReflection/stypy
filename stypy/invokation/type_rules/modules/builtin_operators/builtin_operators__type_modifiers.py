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
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type, get_sample_instance_for_type
from stypy.types.union_type import UnionType

class TypeModifiers:
    @staticmethod
    def eq(localization, proxy_obj, arguments):
        pass