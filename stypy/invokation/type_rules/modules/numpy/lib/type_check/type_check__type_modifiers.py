#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import RealNumber
from stypy.types.type_containers import get_contained_elements_type, \
    can_store_elements


class TypeModifiers:
    @staticmethod
    def iscomplex(localization, proxy_obj, arguments):
        if RealNumber == type(arguments[0]):
            return False

        return call_utilities.create_numpy_array(False)

    @staticmethod
    def nan_to_num(localization, proxy_obj, arguments):
        if RealNumber == type(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])
        if can_store_elements(arguments[0]):
            return call_utilities.create_numpy_array(get_contained_elements_type(arguments[0]))

        return call_utilities.create_numpy_array(arguments[0].wrapped_type)

    @staticmethod
    def real_if_close(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]):
            return call_utilities.cast_to_numpy_type(arguments[0])
        return call_utilities.create_numpy_array(arguments[0], False)
