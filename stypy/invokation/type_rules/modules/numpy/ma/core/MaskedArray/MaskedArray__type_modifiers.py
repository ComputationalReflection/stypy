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
    def mean(localization, proxy_obj, arguments):
        return proxy_obj.im_self.min()

