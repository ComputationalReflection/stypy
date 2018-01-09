#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


class TypeModifiers:
    @staticmethod
    def dirname(localization, proxy, arguments):
        path = arguments[0]
        if path is str():
            return str()

        return os.path.dirname(path)
