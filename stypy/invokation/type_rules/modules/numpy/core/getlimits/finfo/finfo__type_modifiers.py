#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.errors.type_error import StypyTypeError


class TypeModifiers:
    @staticmethod
    def finfo(localization, proxy_obj, arguments):
        if isinstance(type(arguments[0]), type):
            return numpy.core.getlimits.finfo(arguments[0])
        else:
            try:
                return numpy.core.getlimits.finfo(arguments[0])
            except:
                return StypyTypeError(localization, "finfo expects a dtype as a parameter, but {0} was received".format(
                    str(arguments[0])))

