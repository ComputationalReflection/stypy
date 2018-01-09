#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta


class AbstractCallHandler(object):
    __metaclass__ = ABCMeta

    def supports_union_types(self):
        return False

    @abstractmethod
    def can_be_applicable_to(self, callable_):
        pass

    @abstractmethod
    def __call__(self, applicable_rules, localization, callable_, *arguments, **keyword_arguments):
        pass
