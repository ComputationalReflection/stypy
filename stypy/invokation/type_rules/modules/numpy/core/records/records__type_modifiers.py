#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from stypy import contexts
from stypy.errors.advice import Advice
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.invokation.handlers.type_rules_handler import TypeRulesHandler
from stypy.invokation.type_rules.type_groups.type_groups import AnyType, IterableDataStructureWithTypedElements, DynamicType
from stypy.invokation.type_rules.type_groups.type_group_generator import Str
from stypy.module_imports.python_library_modules import is_python_library_module
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance, invoke, python_operator
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper, wrap_contained_type
from stypy.types.type_containers import get_contained_elements_type, set_contained_elements_type, \
    set_contained_elements_type_for_key, can_store_keypairs, can_store_elements, get_key_types
from stypy.types.type_inspection import is_error, is_str, is_function, is_method, dir_object, is_undefined, compare_type
from stypy.types.type_intercession import get_member, set_member, has_member, del_member
from stypy.invokation.handlers import call_utilities

class TypeModifiers:
    @staticmethod
    def fromarrays(localization, proxy, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder'], {
            'dtype': type,
            'shape': AnyType,
            'formats': AnyType,
            'names': [Str, IterableDataStructureWithTypedElements(Str)],
            'titles':   [Str, IterableDataStructureWithTypedElements(Str)],
                                                       'aligned': bool,
                                                       'byteorder': AnyType}, 'fromarrays')

        if isinstance(dvar, StypyTypeError):
            return dvar

        arr = arguments[0]

        contained = get_contained_elements_type(arr)
        if isinstance(contained, union_type.UnionType):
            contained = contained.types

        try:
            return numpy.core.records.fromarrays(contained, dvar)
        except Exception as ex:
            return DynamicType
