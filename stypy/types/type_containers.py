#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import types

import stypy
from stypy.errors.type_error import StypyTypeError
from stypy.reporting.localization import Localization
from stypy.types import undefined_type, union_type
from stypy.types.type_wrapper import TypeWrapper

"""
This file contains functions that deal with types that may contain other types
"""

"""
These container types store its content types inside them
"""
types_that_store_contents_directly = [list, bytearray]#, numpy.ndarray]


def format_type(obj):
    if type(obj) is types.TypeType:
        return obj.__name__
    else:
        return type(obj).__name__


def get_contained_elements_type(proxy_obj):
    """
    Get the type of the contained elements of the passed type
    :param proxy_obj:
    :return:
    """

    # Direct containers
    if type(proxy_obj) in types_that_store_contents_directly:
        try:
            return proxy_obj[0]
        except:
            return undefined_type.UndefinedType

    # Wrappers
    if isinstance(proxy_obj, TypeWrapper):
        if proxy_obj.can_store_elements():
            if type(proxy_obj.wrapped_type) in types_that_store_contents_directly:
                try:
                    return proxy_obj.wrapped_type[0]
                except:
                    return undefined_type.UndefinedType
            return proxy_obj.get_contained_type()
    else:
        # Error: Containers must be wrapped
        if proxy_obj is not None:
            return StypyTypeError(Localization.get_current(),
                                  "Elements of type '{0}' cannot contain types".format(format_type(proxy_obj)))
        else:
            return StypyTypeError(Localization.get_current(),
                                  "The provided type cannot contain elements")


def set_contained_elements_type(proxy_obj, new_type):
    """
    Set the contained elements of a container
    :param proxy_obj:
    :param new_type:
    :return:
    """

    # Direct containers
    if type(proxy_obj) in types_that_store_contents_directly:
        try:
            if len(proxy_obj) > 0:
                return proxy_obj.__setitem__(0, new_type)
            else:
                return proxy_obj.append(new_type)
        except:
            pass

    # Wrappers
    if isinstance(proxy_obj, TypeWrapper):
        if type(proxy_obj.wrapped_type) in types_that_store_contents_directly:
            try:
                if len(proxy_obj.wrapped_type) > 0:
                    return proxy_obj.wrapped_type.__setitem__(0, new_type)
                else:
                    if hasattr(proxy_obj.wrapped_type, 'append'):
                        return proxy_obj.wrapped_type.append(new_type)
            except:
                pass

        return proxy_obj.set_contained_type(new_type)

    return StypyTypeError(Localization.get_current(),
                          "Elements of type '{0}' cannot contain types".format(format_type(proxy_obj)))


def is_type_inference_file_object(proxy_obj):
    """
    Determines if this object is declared in a generated type inference program
    :param proxy_obj:
    :return:
    """
    if hasattr(proxy_obj.__class__, '__module__'):
        if "stypy.sgmc.sgmc_cache" in proxy_obj.__class__.__module__:
            return True

    return False


def get_contained_elements_type_for_key(proxy_obj, key):
    """
    Get the contained element types that correspond to a certain key type (for dictionaries)
    :param proxy_obj:
    :param key:
    :return:
    """
    if isinstance(proxy_obj, union_type.UnionType):
        return proxy_obj.get_contained_type_for_key(key)

    if hasattr(proxy_obj, '__getitem__'):
        try:
            if is_type_inference_file_object(proxy_obj):
                return proxy_obj.__getitem__(Localization.get_current(), key)
            return proxy_obj.__getitem__(key)
        except Exception as exc:
            return undefined_type.UndefinedType
    else:
        return None


def set_contained_elements_type_for_key(proxy_obj, key, new_type):
    """
    Set the contained element types that correspond to a certain key type (for dictionaries)
    :param proxy_obj:
    :param key:
    :param new_type:
    :return:
    """
    if isinstance(proxy_obj, union_type.UnionType):
        return proxy_obj.set_contained_type_for_key(key, new_type)

    if hasattr(proxy_obj, '__setitem__'):
        existing_type = get_contained_elements_type_for_key(proxy_obj, key)
        if existing_type is undefined_type.UndefinedType:
            if proxy_obj is os.environ and key == '':
                return proxy_obj.__setitem__(' ', new_type)
            if is_type_inference_file_object(proxy_obj):
                return proxy_obj.__setitem__(Localization.get_current(), key, new_type)
            return proxy_obj.__setitem__(key, new_type)
        else:
            existing_type = union_type.UnionType.add(existing_type, new_type)
            if is_type_inference_file_object(proxy_obj):
                return proxy_obj.__setitem__(Localization.get_current(), key, existing_type)
            return proxy_obj.__setitem__(key, existing_type)
    else:
        return None


def del_contained_elements_type(proxy_obj, value):
    """
    Deletes the contained element types that correspond to a certain key type (for dictionaries)
    :param proxy_obj:
    :param value:
    :return:
    """
    if can_store_elements(proxy_obj) or can_store_keypairs(proxy_obj):
        try:
            if type(proxy_obj) is TypeWrapper:
                delitem = proxy_obj.get_type_of_member('__delitem__')
                if isinstance(delitem, StypyTypeError):
                    return delitem
                else:
                    return stypy.type_inference_programs.stypy_interface.invoke(Localization.get_current(), delitem,
                                                                                value)
            else:
                if not hasattr(proxy_obj, '__delitem__'):
                    return StypyTypeError(Localization.get_current(),
                                          "Trying to remove elements on a non-container type")
        except:
            return StypyTypeError(Localization.get_current(), "Trying to remove elements on a non-container type")
    else:
        return StypyTypeError(Localization.get_current(), "Trying to remove elements on a non-container type")


def can_store_elements(obj):
    """
    Determines if an object can store elements
    :param obj:
    :return:
    """
    if isinstance(obj, TypeWrapper):
        return obj.can_store_elements()
    return False


def can_store_keypairs(obj):
    """
    Determines if an object can store keypairs
    :param obj:
    :return:
    """
    if isinstance(obj, TypeWrapper):
        return obj.can_store_keypairs()
    else:
        return isinstance(obj, dict) or isinstance(obj, types.DictProxyType) or (type(obj) is types.InstanceType
                                                                                 and hasattr(obj, '__setitem__'))


def get_key_types(obj):
    """
    Get the different key types of a dictionary-like object
    :param obj:
    :return:
    """
    keys = obj.keys()
    return union_type.UnionType.create_from_type_list(keys)


def get_value_types(obj):
    """
    Get the different value types of a dictionary-like object
    :param obj:
    :return:
    """
    values = obj.values()
    temp = None
    for value in values:
        temp = union_type.UnionType.add(temp, value)
    return temp
