#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import sys
import types

from stypy.errors.type_error import StypyTypeError
from stypy.stypy_parameters import special_name_method_prefix
from stypy.types import undefined_type
from stypy.types import union_type
from stypy.types.type_wrapper import TypeWrapper
import stypy

"""
This file contains funcions that provides us introspection capabilities of Python objects.
"""

"""
If a class has these methods defined, its type inference counterpart will change its corresponding method names to
a special form in order to avoid conflicts with Python code behavior.
"""
special_name_methods = ['__eq__', '__cmp__', '__hash__', '__str__', '__repr__']


def is_old_style_class(clazz):
    """
    Determines if a class is an old-style class (don't inherit from object)
    :param clazz:
    :return:
    """
    return not issubclass(clazz, object)


def is_special_name_method(name):
    """
    Determines if the passed method name will be changed in the type inference file method equivalent
    :param name:
    :return:
    """
    return name in special_name_methods


def convert_special_name_method(name):
    """
    Converts a method name to a conflict-free name (use only for special methods)
    :param name:
    :return:
    """
    return special_name_method_prefix + name


def is_class(callable_entity):
    """
    Determines if the passed object is a class
    :param callable_entity:
    :return:
    """
    return type(callable_entity) is types.InstanceType or type(callable_entity) is types.ClassType


def is_function(callable_entity):
    """
    Determines if the passed object is a function
    :param callable_entity:
    :return:
    """
    if inspect.isfunction(callable_entity):
        return True
    return type(callable_entity) in [types.BuiltinFunctionType, types.LambdaType]


def is_method(callable_entity):
    """
    Determines if the passed object is a method
    :param callable_entity:
    :return:
    """
    try:
        defining_class = get_defining_class(callable_entity)
        return inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity) or \
               type(callable_entity).__name__ == 'method-wrapper' or (
                   inspect.isbuiltin(callable_entity) and defining_class not in [None, types.NoneType])
    except:
        return False


def get_defining_class(method):
    """
    Gets the class that defines the passed method
    :param method:
    :return:
    """
    if hasattr(method, "im_class"):
        if method.im_class is None:
            if hasattr(method, "im_self"):
                if method.im_self is not None:
                    return type(method.im_self)
        else:
            if method.im_class.__name__ == "ABCMeta":
                if hasattr(method, "im_self"):
                    if method.im_self is not None:
                        return method.im_self

        return method.im_class
    if hasattr(method, "__objclass__"):
        return method.__objclass__
    if hasattr(method, "__self__"):
        return type(method.__self__)
    if hasattr(method, "im_self"):
        return type(method.im_self)
    return None


def get_self(method):
    """
    Get the self parameter of a method
    :param method:
    :return:
    """
    if hasattr(method, "__self__"):
        return method.__self__
    if hasattr(method, "__objclass__"):
        return method.__objclass__
    return None


def get_name(obj):
    """
    Get the name of a Python object
    :param obj:
    :return:
    """
    if hasattr(obj, "__name__"):
        return obj.__name__

    if hasattr(type(obj), "__name__"):
        return type(obj).__name__

    return None


def get_defining_module(obj):
    """
    Gets the module that defined the passed object
    :param obj:
    :return:
    """
    if isinstance(obj, str):
        return obj  # Already a module name

    if hasattr(obj, "__module__"):
        return obj.__module__

    return None


def is_pyd_module(module_info):
    """
    Determines if the passed module is a .pyd module
    :param module_info:
    :return:
    """
    try:
        if type(module_info) is str:
            mod = sys.modules[module_info]
        else:
            mod = module_info

        return mod.__file__.endswith(".pyd")
    except:
        return False


def is_str(type_):
    """
    Determines if the passed object is a str
    :param type_:
    :return:
    """
    return type_ is str or type_ is unicode


def is_error(obj):
    """
    Determines if the passed object is a stypy type error
    :param obj:
    :return:
    """
    return isinstance(obj, StypyTypeError)


def is_undefined(obj):
    """
    Determines if the passed object is an undefined type
    :param obj:
    :return:
    """
    return isinstance(obj, undefined_type.UndefinedType) or obj is undefined_type.UndefinedType


def is_union_type(obj):
    """
    Determines if the passed object is a union type
    :param obj:
    :return:
    """
    return isinstance(obj, union_type.UnionType)


def compare_type(obj1, obj2):
    """
    Compares two types using its values
    :param obj1:
    :param obj2:
    :return:
    """
    if isinstance(obj1, union_type.UnionType):
        for t in obj1.types:
            if isinstance(t, TypeWrapper):
                t = t.get_wrapped_type()
            if isinstance(obj2, TypeWrapper):
                obj2 = type(obj2.get_wrapped_type())
            if not type(t) is obj2:
                return False

        return True
    if isinstance(obj1, TypeWrapper):
        obj1 = obj1.get_wrapped_type()
    if isinstance(obj2, TypeWrapper):
        obj2 = type(obj2.get_wrapped_type())

    return type(obj1) is obj2


def dir_object(obj):
    """
    Obtain the result of calling the dir (...) primitive passing the received object
    :param obj:
    :return:
    """
    if isinstance(obj, TypeWrapper):
        return dir(obj.wrapped_type)
    return dir(obj)


def is_recursive_call_result(obj):
    return type(obj) is stypy.types.no_recursion.RecursionType or obj is stypy.types.no_recursion.RecursionType

