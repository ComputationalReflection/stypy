#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseTypeGroup(object):
    """
    All type groups inherit from this class
    """

    def __str__(self):
        return self.__repr__()


class TypeGroup(BaseTypeGroup):
    """
    A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups
    are collections of types that have something in common, and Python functions and methods usually admits any of them
    as a parameter when one of them is valid. For example, if a Python library function works with an int as the first
    parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a
    TypeGroup that will be called Integer

    Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with
    a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python
    object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches
    with classes that are a subtype of the one specified.

    Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and
    flexibility to specify admitted types in any Python callable entity.

    Type groups are created in the file type_groups.py
    """

    def __init__(self, grouped_types):
        """
        Create a new type group that represent the list of types passed as a parameter
        :param grouped_types: List of types that are included inside this type group
        :return:
        """
        self.grouped_types = grouped_types

    def __contains__(self, type_):
        """
        Test if this type group contains the specified type (in operator)
        :param type_: Type to test
        :return: bool
        """
        return type_ in self.grouped_types

    def __eq__(self, type_):
        """
        Test if this type group contains the specified type (== operator)
        :param type_: Type to test
        :return: bool
        """
        return type_ in self.grouped_types

    def __cmp__(self, type_):
        """
        Test if this type group contains the specified type (compatarion operators)
        :param type_: Type to test
        :return: bool
        """
        return type_ in self.grouped_types

    def __gt__(self, other):
        """
        Type group sorting. A type group is less than other type group if contains less types or the types contained
        in the type group are all contained in the other one. Otherwise, is greater than the other type group.
        :param other: Another type group
        :return: bool
        """
        if len(self.grouped_types) < len(other.grouped_types):
            return False

        for type_ in self.grouped_types:
            if type_ not in other.grouped_types:
                return False

        return True

    def __lt__(self, other):
        """
        Type group sorting. A type group is less than other type group if contains less types or the types contained
        in the type group are all contained in the other one. Otherwise, is greater than the other type group.
        :param other: Another type group
        :return: bool
        """
        if len(self.grouped_types) > len(other.grouped_types):
            return False

        for type_ in self.grouped_types:
            if type_ not in other.grouped_types:
                return False

        return True

    def __repr__(self):
        """
        Textual representation of the type group
        :return: str
        """
        ret_str = type(self).__name__
        return ret_str
