#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


class TypeModifiers:
    @staticmethod
    def compile(localization, proxy, arguments):
        import re
        return re.compile(arguments[0])
