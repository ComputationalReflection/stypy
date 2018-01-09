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


class TypeModifiers:
    @staticmethod
    def lookfor(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments, ['module', 'import_modules',
                                                                                 'regenerate', 'output'], {
            'module': [Str, IterableDataStructureWithTypedElements(Str)],
            'import_modules': bool,
            'regenerate': bool,
            'output': file,
        }, 'lookfor')

        if isinstance(dvar, StypyTypeError):
            return dvar

        return types.NoneType