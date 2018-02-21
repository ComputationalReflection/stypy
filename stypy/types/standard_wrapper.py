#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy.types.type_containers import get_contained_elements_type, can_store_elements, can_store_keypairs, \
    get_key_types, get_contained_elements_type_for_key
from stypy.types.type_inspection import is_union_type, is_undefined
from type_wrapper import TypeWrapper
import numpy
import itertools
import collections

types_to_be_wrapped = [list, dict, numpy.ndarray, itertools.product, collections.deque]


def wrap_contained_type(type_):
    """
    Wraps a type if the type need to be wrapped
    :param type_:
    :return:
    """
    if type(type_) in types_to_be_wrapped or __can_store_elements(type(type_)):
        return StandardWrapper(type_)

    for t in types_to_be_wrapped:
        if issubclass(type(type_), t):
            return StandardWrapper(type_)

    return type_


def __can_store_elements(python_type):
    """
    Determines if this proxy represents a Python type able to store elements (lists, tuples, ...)
    :return: bool
    """
    name = python_type.__name__
    is_iterator = ("dictionary-" in name and "iterator" in name) or ("iterator" in name and
                                                                     "dict" not in name)

    data_structures = [set, tuple, types.GeneratorType, slice, range, xrange, enumerate, reversed,
                       frozenset]

    data_structure = filter(lambda ds: python_type is ds, data_structures)
    return len(data_structure) > 0 or is_iterator


class StandardWrapper(TypeWrapper):
    """
    Class to wrap certain types for hash functionality and contained types control. It overloads most Python operators
    to redirect its invocation over the wrapper object to the wrapped object, so users have not to change its code to
    substitute the wrapped for the wrapper in most situations.
    """
    declared_members = None

    def __get_wrapped_type_member(self, member_name):
        """
        Gets a member from the wrapped type
        :param member_name:
        :return:
        """
        if hasattr(self.wrapped_type, member_name):
            return getattr(self.wrapped_type, member_name)
        else:
            return getattr(type(self.wrapped_type), member_name)

    def __init__(self, wrapped_type):
        """
        Initializes the wrapper with the wrapped type
        :param wrapped_type:
        """
        TypeWrapper.__init__(self, wrapped_type)
        self.__module__ = self.__get_wrapped_type_member('__module__')
        self.__name__ = self.__get_wrapped_type_member('__name__')

    def is_declared_member(self, name):
        """
        Determines if a member has been declared on the wrapped type
        :param name:
        :return:
        """
        return name in self.declared_members

    def __len__(self):
        """
        len operator overload
        :return:
        """
        len_ = object.__getattribute__(self.wrapped_type, '__len__')
        return len_()

    def __getitem__(self, item_):
        """
        [] operator overload
        :return:
        """
        get_item = object.__getattribute__(self.wrapped_type, '__getitem__')
        return wrap_contained_type(get_item(item_))

    def __setitem__(self, item_, value):
        """
        [] operator overload (write)
        :return:
        """
        set_item = object.__getattribute__(self.wrapped_type, '__setitem__')
        return set_item(item_, value)

    def get_type_of_member(self, name):
        """
        Gets the type of a member of the wrapped type
        :param name:
        :return:
        """
        if name in self.overriden_members:
            return wrap_contained_type(object.__getattribute__(self, name))
        return wrap_contained_type(object.__getattribute__(self.wrapped_type, name))

        #return wrap_contained_type(object.__getattribute__(object.__getattribute__(self, 'wrapped_type'), name))

    def __getattribute__(self, name):
        """
            Equivalent to get_type_of_member
        """
        if name in object.__getattribute__(self, "declared_members"):
            return object.__getattribute__(self, name)
        else:
            return self.get_type_of_member(name)

    def set_type_of_member(self, name, value):
        """
        Sets the type of a member of the wrapped type
        :param name:
        :param value:
        :return:
        """
        if name in self.overriden_members:
            return object.__setattr__(self, name, value)
        return object.__setattr__(self.wrapped_type, name, value)

    def del_member(self, name):
        """
        Deletes a member of the wrapped type
        :param name:
        :return:
        """
        return object.__delattr__(self.wrapped_type, name)

    def __setattr__(self, name, value):
        """
        Equivalent to set_type_of_member
        :param name:
        :param value:
        :return:
        """
        if name in object.__getattribute__(self, "declared_members"):
            return object.__setattr__(self, name, value)
        else:
            return self.set_type_of_member(name, value)

    def __cmp__(self, other):
        """
        == operator overload
        :param other:
        :return:
        """
        return self.wrapped_type.__eq__(other)

    def __eq__(self, other):
        """
        == operator overload
        :param other:
        :return:
        """
        if not isinstance(other, TypeWrapper):
            return False

        if "ndarray" in type(self.wrapped_type).__name__ and "ndarray" in type(other.wrapped_type).__name__:
            wrapped_eq = self.wrapped_type.tolist() == other.wrapped_type.tolist()
        else:
            wrapped_eq = self.wrapped_type == other.wrapped_type

            # Special comparison for tuples
            if not wrapped_eq and type(self.wrapped_type) is tuple and type(other.wrapped_type) is tuple:
                t1 = []
                for e in self.wrapped_type:
                    t1.append(type(e))

                t2 = []
                for e in other.wrapped_type:
                    t2.append(type(e))
                wrapped_eq = t1 == t2

        if not wrapped_eq:
            return False

        if hasattr(self, 'contained_types'):
            contained_1 = self.contained_types
        else:
            contained_1 = None

        if hasattr(other, 'contained_types'):
            contained_2 = other.contained_types
        else:
            contained_2 = None

        return contained_1 == contained_2

    def __repr__(self):
        """
        str operator overload
        :return:
        """
        txt = type(self.wrapped_type).__name__
        if can_store_keypairs(self):
            txt += "{"
            keys = get_key_types(self)
            if is_union_type(keys):
                keys = keys.get_types()
            else:
                keys = list(keys)

            if len(keys) == 0:
                txt += "UndefinedType"
            else:
                for key in keys:
                    values = get_contained_elements_type_for_key(self, key)
                    if not isinstance(values, TypeWrapper):
                        contents = type(values).__name__
                    else:
                        contents = str(values)
                    txt += type(key).__name__ + ": " + contents + "; "
                txt = txt[:-2]
            return txt + "}\n"

        if can_store_elements(self):
            contained_type = get_contained_elements_type(self)
            if not isinstance(contained_type, TypeWrapper):
                if is_undefined(contained_type):
                    contents = "UndefinedType"
                else:
                    contents = type(contained_type).__name__
            else:
                contents = str(contained_type)
            return txt + "[" + contents + "]"

        return txt


"""
When obtaining members from a wrapper, certain members are from the wrapper itself instead of the wrapped object. This
contains a list of members of the wrapper so member get and set operations may handle its calls properly.
"""
if StandardWrapper.declared_members is None:
    StandardWrapper.declared_members = [item for item in dir(StandardWrapper) if item not in dir(object)] + \
                                       ["wrapped_type", "__hash__", "__cmp__", "__get_wrapped_type_member",
                                        "__name__", "contained_types", "__repr__", "__class__", "overriden_members"]
