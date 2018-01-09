

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, RealNumber, \
    IterableDataStructureWithTypedElements, IterableDataStructure

from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type

class TypeModifiers:
    @staticmethod
    def __getitem__(localization, proxy_obj, arguments):
        if Integer == type(arguments[0]):
            return int()
        else:
            t = call_utilities.wrap_contained_type(tuple())
            set_contained_elements_type(localization, t, int())

            return t


