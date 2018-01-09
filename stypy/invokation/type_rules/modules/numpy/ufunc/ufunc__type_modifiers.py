
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types, numpy

from stypy.types.type_wrapper import TypeWrapper
from stypy.types.union_type import UnionType
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Number, Integer
from stypy.type_inference_programs.stypy_interface import get_contained_elements_type, set_contained_elements_type
from stypy.errors.type_error import StypyTypeError

class TypeModifiers:
    @staticmethod
    def reduceat(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['axis', 'dtype', 'out'], {
            'axis': Integer,
            'dtype': type,
            'out': numpy.ndarray,
            }, 'reduceat', 2)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'out' in dvar.keys():
            set_contained_elements_type(localization, dvar['out'], get_contained_elements_type(localization, arguments[0]))
            return dvar['out']

        return arguments[0]
