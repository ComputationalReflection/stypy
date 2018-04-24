#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import types

from stypy.errors.type_error import StypyTypeError
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance, wrap_contained_type
from stypy.types import undefined_type
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.type_containers import get_contained_elements_type, \
    set_contained_elements_type
from stypy.types.type_inspection import is_method, is_function, is_class, get_self, is_str, compare_type


class TypeModifiers:
    # ###################### LIST TYPE MODIFIERS #######################

    @staticmethod  # Constructor  (__init__)
    def difference(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'set')
        ret_type = wrap_contained_type(ret_type)

        my_type = StandardWrapper.get_wrapper_of(proxy_obj)
        my_contained_type = get_contained_elements_type(my_type)
        other_contained_type = get_contained_elements_type(arguments[0])

        typ = union_type.UnionType.add(other_contained_type, my_contained_type)

        set_contained_elements_type(ret_type, typ)

        return ret_type

    @staticmethod
    def add(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        existing_type = get_contained_elements_type(self_instance)
        if existing_type is undefined_type.UndefinedType:
            new_type = arguments[0]
        else:
            new_type = union_type.UnionType.add(existing_type, arguments[0])
        set_contained_elements_type(self_instance, new_type)
        return types.NoneType

    @staticmethod
    def __getitem__(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        return get_contained_elements_type(self_instance)

    @staticmethod
    def pop(localization, callable_, arguments):
        self_instance = StandardWrapper.get_wrapper_of(callable_.__self__)
        return get_contained_elements_type(self_instance)