#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy.errors.advice import Advice
from stypy.errors.type_error import StypyTypeError
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types import union_type
from stypy.types.type_inspection import get_self
from stypy.types.type_intercession import get_member, set_member, del_member


class TypeModifiers:
    @staticmethod
    def stypy____repr__(localization, proxy_obj, arguments):
        return get_builtin_python_type_instance(localization, "str")

    @staticmethod
    def stypy____str__(localization, proxy_obj, arguments):
        return get_builtin_python_type_instance(localization, "str")

    @staticmethod
    def stypy____getattribute__(localization, proxy_obj, arguments):
        param = arguments[0]

        if not param is str():
            return get_member(localization, get_self(proxy_obj), param)
        else:
            members = dir(get_self(proxy_obj))
            ret_type = None
            for member in members:
                member_type = get_member(localization, get_self(proxy_obj), member)
                ret_type = union_type.UnionType.add(ret_type, member_type)
            return ret_type

    @staticmethod
    def stypy____setattr__(localization, proxy_obj, arguments):
        attr_name = arguments[0]
        attr_value = arguments[1]

        if attr_name is str():
            Advice(localization,
                   "Called __setattr__ without a value in the member name parameter: the operation cannot be checked")
            return types.NoneType

        return set_member(localization, get_self(proxy_obj), attr_name, attr_value)

    @staticmethod
    def stypy____delattr__(localization, proxy_obj, arguments):
        attr_name = arguments[0]

        if attr_name is str():
            Advice(localization,
                   "Called __delattr__ without a value in the member name parameter: the operation cannot be checked")
            return types.NoneType
        try:
            del_member(localization, get_self(proxy_obj), attr_name)
        except Exception as exc:
            return StypyTypeError.member_cannot_be_deleted_error(localization, get_self(proxy_obj), attr_name, str(exc))

        return types.NoneType
