#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types import union_type
from stypy.types.type_containers import set_contained_elements_type_for_key, \
    get_key_types, set_contained_elements_type, get_contained_elements_type, can_store_keypairs


class TypeModifiers:
    @staticmethod
    def fromkeys(localization, proxy_obj, arguments):
        if len(arguments) > 0:
            param1 = arguments[0]
        else:
            param1 = types.NoneType

        if len(arguments) > 1:
            param2 = arguments[1]
        else:
            param2 = types.NoneType

        ret = get_builtin_python_type_instance(localization, "dict")

        # There are several cases:
        # A dictionary: Return a copy
        # A dictionary and any other object: {<each dict key>: other object}
        if can_store_keypairs(param1):
            if param2 == types.NoneType:
                return param1
            else:
                temp = param1
                set_contained_elements_type(temp, get_builtin_python_type_instance(localization, "dict"))
                keys = get_key_types(param1)
                if isinstance(keys, union_type.UnionType):
                    keys = keys.types
                else:
                    keys = [keys]
                for key in keys:
                    set_contained_elements_type_for_key(temp, key, param2)

                return temp
        else:
            # A list or a tuple: {each structure element type: None}
            # A list or a tuple and any other object: {<each dict key>: other object}
            t1 = get_contained_elements_type(param1)
            if isinstance(t1, union_type.UnionType):
                t1 = t1.types
            else:
                t1 = [t1]

            if param2 == types.NoneType:
                value = types.NoneType
            else:
                value = param2

            for t in t1:
                set_contained_elements_type_for_key(ret, t, value)

        return ret
