#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.types.type_containers import get_contained_elements_type
from stypy.types.type_inspection import get_self
from stypy.types.type_wrapper import TypeWrapper


class TypeModifiers:
    @staticmethod
    def next(localization, proxy_obj, arguments):
        self_object = get_self(proxy_obj)

        return get_contained_elements_type(TypeWrapper.get_wrapper_of(self_object))
