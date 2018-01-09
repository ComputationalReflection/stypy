#!/usr/bin/env python
# -*- coding: utf-8 -*-


class UndefinedType(object):
    """
    A type that identifies those situations in which no type can be inferred.
    """
    def __repr__(self):
        return "UndefinedType"

    def __str__(self):
        return self.__repr__()
