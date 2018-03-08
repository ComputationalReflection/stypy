#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from stypy.types.undefined_type import UndefinedType


class TypeWrapper(object):
    """
    Base class of any type wrapper
    """
    __metaclass__ = ABCMeta
    all_wrappers = dict()

    def __init__(self, wrapped_type):
        """
        Initializes the wrapper to wrap the passed type
        :param wrapped_type:
        """
        self.wrapped_type = wrapped_type
        self.overriden_members = []
        TypeWrapper.all_wrappers[self] = wrapped_type

    @staticmethod
    def get_wrapper_of(obj):
        """
        There is a wrapper registry that contains all wrappers created so far. This is the only mechanism that we
        have to obtain the wrapper of a certain Python object, as it is not possible to modify the object to establish
        a bidirectional association between them
        :param obj:
        :return:
        """
        for key, value in TypeWrapper.all_wrappers.iteritems():
            if value is obj:
                return key
        return None

    def get_wrapped_type(self):
        """
        Gets the wrapped type
        :return:
        """
        return self.wrapped_type

    def can_store_elements(self):
        """
        Determines if this wrapper can store elements
        :return:
        """
        return hasattr(self, 'contained_types') or hasattr(self.wrapped_type, '__getitem__')

    def can_store_keypairs(self):
        """
        Determines if this wrapper can store key pairs
        :return:
        """
        return isinstance(self.wrapped_type, dict)

    def get_contained_type(self, multi_assign_arity=-1, multi_assign_index=-1):
        """
        Gets the types contained by the wrapped type
        :return:
        """
        if hasattr(self, 'contained_types'):
            if type(self.wrapped_type) is tuple:
                # Tuples in certain multiple assignments return individual components instead of the whole tuple
                try:
                    if isinstance(self.contained_types, TypeWrapper):
                        if len(self.contained_types.types) == multi_assign_arity:
                            return self.contained_types.types[multi_assign_index]
                except:
                    pass

            return self.contained_types
        else:
            return UndefinedType

    def set_contained_type(self, type_):
        """
        Sets the types contained by the wrapped type
        :param type_:
        :return:
        """
        self.contained_types = type_

    # Abstract methods that deal with members of the wrapped types, to be defined by subclasses.

    @abstractmethod
    def is_declared_member(self, name):
        pass

    @abstractmethod
    def get_type_of_member(self, name):
        pass

    @abstractmethod
    def set_type_of_member(self, name, value):
        pass

    @abstractmethod
    def del_member(self, name):
        pass
