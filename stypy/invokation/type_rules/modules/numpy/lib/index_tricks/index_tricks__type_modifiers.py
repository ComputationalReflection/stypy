#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type
from stypy.types import union_type

class TypeModifiers:
    @staticmethod
    def ndindex(localization, proxy_obj, arguments):
        if Number == type(call_utilities.get_contained_elements_type(localization, arguments[0])):
            import numpy
            return numpy.lib.ndindex(3)

        return StypyTypeError(localization, "Invalid type for ndindex argument")

