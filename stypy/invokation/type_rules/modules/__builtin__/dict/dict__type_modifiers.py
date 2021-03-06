#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy.errors.type_error import StypyTypeError
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type_instance
from stypy.types import union_type
from stypy.types.standard_wrapper import StandardWrapper
from stypy.types.type_containers import can_store_keypairs, get_contained_elements_type
from stypy.types.type_containers import set_contained_elements_type_for_key, \
    get_contained_elements_type_for_key, get_key_types, get_value_types, set_contained_elements_type
from stypy.types.type_inspection import compare_type, get_self, is_error
from stypy.types.undefined_type import UndefinedType


class TypeModifiers:
    @staticmethod
    def dict(localization, proxy_obj, arguments):
        ret = get_builtin_python_type_instance(localization, 'dict')
        if len(arguments) == 0:
            return ret
        if can_store_keypairs(arguments[0]):
            if compare_type(arguments[0], types.DictProxyType):
                if isinstance(arguments[0], StandardWrapper):
                    contents = arguments[0].get_wrapped_type()
                else:
                    contents = arguments[0]
                ret = dict(contents)
                ret = StandardWrapper(ret)
                return ret

            return arguments[0]
        else:
            contents = get_contained_elements_type(arguments[0])
            if not compare_type(contents, tuple):
                return StypyTypeError.object_must_be_type_error(localization,
                                                                'Iterable argument to build a dictionary',
                                                                '(key,value) tuple')
            else:
                keys = get_contained_elements_type(contents)
                values = keys
                if isinstance(keys, union_type.UnionType):
                    keys = keys.types
                else:
                    keys = [keys]

                if len(keys) == 2:
                    # Special case in which we use a two-element tuple to initialize a dict. We assume that the first
                    # element is the key and the second the value
                    set_contained_elements_type_for_key(ret, keys[0], keys[1])
                else:
                    for key in keys:
                        set_contained_elements_type_for_key(ret, key, values)

                return ret

    ####### TYPE MODIFIERS #######

    @staticmethod
    def __getitem__(localization, proxy_obj, arguments):
        if type(arguments[0]) is str:
            # string with a concrete value
            if arguments[0] is not str():
                ret = get_contained_elements_type_for_key(get_self(proxy_obj), arguments[0])
                if ret is UndefinedType:
                    ret = get_contained_elements_type_for_key(get_self(proxy_obj), str())
                    if ret is UndefinedType:
                        return StypyTypeError(localization, "No elements exist for the key {0}".format(arguments[0]))
                    else:
                        return ret
                else:
                    return ret
            else:
                # String with an undefined value
                self_obj = get_self(proxy_obj)
                union_ret = None
                for key in self_obj.keys():
                    if key is arguments[0] or type(arguments[0]) is type(key):
                        union_ret = union_type.UnionType.add(union_ret,
                                                             get_contained_elements_type_for_key(self_obj, key))

                return union_ret

        ret = get_contained_elements_type_for_key(get_self(proxy_obj), arguments[0])
        if ret is UndefinedType:
            return StypyTypeError.key_error(localization, arguments[0])
        return ret

    @staticmethod
    def clear(localization, proxy_obj, arguments):
        return get_self(proxy_obj).clear()

    @staticmethod
    def __setitem__(localization, proxy_obj, arguments):
        set_contained_elements_type_for_key(get_self(proxy_obj), arguments[0], arguments[1])
        return types.NoneType

    @staticmethod
    def copy(localization, proxy_obj, arguments):
        return get_self(proxy_obj).copy()

    @staticmethod
    def get(localization, proxy_obj, arguments):
        ret = get_contained_elements_type_for_key(get_self(proxy_obj), arguments[0])
        if ret is UndefinedType:
            if len(arguments) > 1:
                return arguments[1]
            else:
                return types.NoneType

        return ret

    @staticmethod
    def __iter__(localization, proxy_obj, arguments):
        ret_type = StandardWrapper(get_self(proxy_obj).__iter__())

        key_list = get_key_types(get_self(proxy_obj))
        stored_keys_type = None
        if isinstance(key_list, union_type.UnionType):
            key_list = key_list.types
        else:
            key_list = list(key_list)

        for key in key_list:
            stored_keys_type = union_type.UnionType.add(stored_keys_type, key)

        if stored_keys_type is not None:
            set_contained_elements_type(ret_type, stored_keys_type)
        else:
            set_contained_elements_type(ret_type, UndefinedType)

        return ret_type

    @staticmethod
    def iterkeys(localization, proxy_obj, arguments):
        return TypeModifiers.__iter__(localization, proxy_obj, arguments)

    @staticmethod
    def itervalues(localization, proxy_obj, arguments):
        ret_type = StandardWrapper(get_self(proxy_obj).itervalues())

        stored_values_type = get_value_types(get_self(proxy_obj))

        if stored_values_type is not None:
            set_contained_elements_type(ret_type, stored_values_type)
        else:
            set_contained_elements_type(ret_type, UndefinedType)

        return ret_type

    @staticmethod
    def iteritems(localization, proxy_obj, arguments):
        ret_type = StandardWrapper(get_self(proxy_obj).iteritems())

        key_types = get_contained_elements_type(TypeModifiers.iterkeys(localization, proxy_obj, arguments))
        value_types = get_contained_elements_type(TypeModifiers.itervalues(localization, proxy_obj, arguments))

        container_type = get_builtin_python_type_instance(localization, "tuple")
        union = union_type.UnionType.add(key_types, value_types)
        set_contained_elements_type(container_type, union)

        set_contained_elements_type(ret_type, container_type)

        return ret_type

    @staticmethod
    def values(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')

        stored_values_type = get_value_types(get_self(proxy_obj))

        if stored_values_type is not None:
            set_contained_elements_type(ret_type, stored_values_type)
        else:
            set_contained_elements_type(ret_type, UndefinedType)

        return ret_type

    @staticmethod
    def keys(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')

        if len(get_self(proxy_obj)) == 0:
            return ret_type

        key_list = get_key_types(get_self(proxy_obj))
        if isinstance(key_list, union_type.UnionType):
            key_list = key_list.types
        else:
            key_list = list(key_list)

        stored_keys_type = None
        for value in key_list:
            stored_keys_type = union_type.UnionType.add(stored_keys_type, value)

        if stored_keys_type is not None:
            set_contained_elements_type(ret_type, stored_keys_type)
        else:
            set_contained_elements_type(ret_type, UndefinedType)

        return ret_type

    @staticmethod
    def items(localization, proxy_obj, arguments):
        ret_type = get_builtin_python_type_instance(localization, 'list')

        key_types = get_contained_elements_type(TypeModifiers.iterkeys(localization, proxy_obj, arguments))
        value_types = get_contained_elements_type(TypeModifiers.itervalues(localization, proxy_obj, arguments))

        container_type = get_builtin_python_type_instance(localization, "tuple")

        union = union_type.UnionType.add(key_types, value_types)
        set_contained_elements_type(container_type, union)

        set_contained_elements_type(ret_type, container_type)

        return ret_type

    @staticmethod
    def popitem(localization, proxy_obj, arguments):
        key_types = get_contained_elements_type(TypeModifiers.iterkeys(localization, proxy_obj, arguments))
        value_types = get_contained_elements_type(TypeModifiers.itervalues(localization, proxy_obj, arguments))

        container_type = get_builtin_python_type_instance(localization, "tuple")
        union = union_type.UnionType.add(key_types, value_types)
        set_contained_elements_type(container_type, union)

        return container_type

    @staticmethod
    def update(localization, proxy_obj, arguments):
        ret = get_self(proxy_obj)
        param = arguments[0]

        if can_store_keypairs(param):
            keys = get_key_types(param)
            if isinstance(keys, union_type.UnionType):
                keys = keys.types
            else:
                keys = list(keys)
            for key in keys:
                value = get_contained_elements_type_for_key(param, key)
                set_contained_elements_type_for_key(ret, key, value)
        else:
            if param.can_store_elements():
                contents = get_contained_elements_type(param)
                if isinstance(contents, tuple):
                    keys = get_contained_elements_type(contents)
                    values = get_contained_elements_type(contents)
                    for key in keys:
                        set_contained_elements_type_for_key(ret, key, values)
                else:
                    return StypyTypeError.invalid_length_error(localization,
                                                               "Dictionary 'update' sequence", 1, 2)
            else:
                return StypyTypeError.object_must_be_type_error(
                    localization,
                    "The 'update' method second parameter", "dict or an iterable object")

        return ret

    @staticmethod
    def setdefault(localization, proxy_obj, arguments):
        ret_type = TypeModifiers.get(localization, proxy_obj, arguments)
        if len(arguments) > 1:
            t2 = arguments[1]
        else:
            t2 = types.NoneType

        # Type do not exist
        if is_error(ret_type):
            t1 = arguments[0]
            set_contained_elements_type_for_key(get_self(proxy_obj), t1, t2)
        else:
            if len(arguments) > 1:
                t1 = arguments[0]
                set_contained_elements_type_for_key(get_self(proxy_obj), t1, t2)
                t2 = union_type.UnionType.add(ret_type, t2)
            else:
                t2 = ret_type

        return t2

    @staticmethod
    def pop(localization, proxy_obj, arguments):
        return get_value_types(get_self(proxy_obj))
