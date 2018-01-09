#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from stypy.invokation.handlers import call_utilities


class TypeModifiers:
    @staticmethod
    def fix(localization, proxy_obj, arguments):
        if call_utilities.is_numpy_array(arguments[0]):
            return arguments[0]
        else:
            return call_utilities.create_numpy_array(arguments[0])
