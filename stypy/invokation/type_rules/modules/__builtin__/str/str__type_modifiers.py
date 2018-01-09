#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types.type_containers import set_contained_elements_type


class TypeModifiers:
    @staticmethod
    def rpartition(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'tuple')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def partition(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'tuple')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def splitlines(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def rsplit(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def split(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def __getnewargs__(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'tuple')
        set_contained_elements_type(ret_type, get_builtin_python_type_instance(localization, 'str'))

        return ret_type

    @staticmethod
    def _formatter_field_name_split(localization, proxy_obj, arguments):
        result = "test"._formatter_field_name_split()
        ret_type = get_builtin_python_type_instance(localization, "tuple")
        set_contained_elements_type(ret_type, result[1])

        return ret_type
