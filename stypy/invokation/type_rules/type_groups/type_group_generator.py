#!/usr/bin/env python
# -*- coding: utf-8 -*-
from type_groups import *


class RuleGroupGenerator(object):
    """
    This class is used to generate type group instances from lists of types defined in the TypeGroup object of the
    type_groups.py file, using the TypeGroup class as a canvas class to generate them.
    """
    rule_group_cache = dict()

    def create_rule_group_class(self, class_name):
        """
        Creates a new class named as class_name, with all the members of the TypeGroup class
        :param class_name: Name of the new class
        :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can
        only be created once. If we try to create one that has been already created, the created one is returned instead
        """
        if class_name in self.rule_group_cache.keys():
            return self.rule_group_cache[class_name]

        group_class = type(class_name, TypeGroup.__bases__, dict(TypeGroup.__dict__))
        instance = group_class(getattr(TypeGroups, class_name))
        self.rule_group_cache[class_name] = instance

        return instance

    def create_rule_group_class_list(self, classes_name):
        """
        Mass-creation of rule group classes calling the previous method
        :param classes_name: List of class names
        :return: List of classes
        """
        instances = []
        for class_name in classes_name:
            instance = self.create_rule_group_class(class_name)
            instances.append(instance)

        return instances

    def __init__(self):
        self.rule_group_compliance_dict = dict()
        for rule in TypeGroups.get_rule_groups():
            self.rule_group_compliance_dict[rule] = [False] * eval("len(TypeGroups.{0})".format(rule))
        self.added_types = []
        self.unclassified_types = []


"""
TypeGroups composed by collections of types
"""
RealNumber = RuleGroupGenerator().create_rule_group_class("RealNumber")
NumpyRealNumber = RuleGroupGenerator().create_rule_group_class("NumpyRealNumber")
Number = RuleGroupGenerator().create_rule_group_class("Number")
NumpyNumber = RuleGroupGenerator().create_rule_group_class("NumpyNumber")
Integer = RuleGroupGenerator().create_rule_group_class("Integer")
NumpyInteger = RuleGroupGenerator().create_rule_group_class("NumpyInteger")
Str = RuleGroupGenerator().create_rule_group_class("Str")
IterableDataStructure = RuleGroupGenerator().create_rule_group_class("IterableDataStructure")
IterableObject = RuleGroupGenerator().create_rule_group_class("IterableObject")
ByteSequence = RuleGroupGenerator().create_rule_group_class("ByteSequence")
