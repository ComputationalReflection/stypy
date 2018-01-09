#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data_in_files_handler import DataInFilesHandler
from stypy import stypy_parameters

special_modifier_methods = ['__getattribute__', '__setattr__', '__delattr__', '__repr__', '__str__']


def rule_getter_function(module, entity_name):
    if entity_name in special_modifier_methods:
        entity_name = "stypy__" + entity_name

    return getattr(module.TypeModifiers, entity_name)


class TypeRulesModifier(DataInFilesHandler):
    def __init__(self):
        rule_getter = rule_getter_function
        super(TypeRulesModifier, self).__init__(stypy_parameters.type_modifier_file_postfix, rule_getter)

    def __call__(self, applicable_modifier, localization, callable_, *arguments, **keyword_arguments):
        return applicable_modifier(localization, callable_, arguments)
