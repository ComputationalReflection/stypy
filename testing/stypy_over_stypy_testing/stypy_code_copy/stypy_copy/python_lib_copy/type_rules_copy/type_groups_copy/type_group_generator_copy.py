from type_groups_copy import *


class RuleGroupGenerator:
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
        # self.rule_group_compliance_dict = dict()
        # for rule in TypeGroups.get_rule_groups():
        #     self.rule_group_compliance_dict[rule] = [False] * eval("len(TypeGroups.{0})".format(rule))
        self.added_types = []
        self.unclassified_types = []

    #TODO: Delete?
    # def add_type(self, type_):
    #     if not type_ in self.added_types:
    #         self.added_types.append(type_)
    #
    #     added = False
    #     for rule_group in TypeGroups.get_rule_groups():
    #         type_list = getattr(TypeGroups, rule_group)
    #         if type_ in type_list:
    #             for i in range(len(type_list)):
    #                 if type_ == type_list[i]:
    #                     self.rule_group_compliance_dict[rule_group][i] = True
    #                     added = True
    #
    #     if not added:
    #         if not type_ in self.unclassified_types:
    #             self.unclassified_types.append(type_)
    #
    # def get_rule_group(self):
    #     ret_rule_group = None
    #     added = False
    #
    #     for (marked_rule_group, marks) in self.rule_group_compliance_dict.items():
    #         true_marks = len(filter(lambda x: x == True, marks))
    #         if len(getattr(TypeGroups, marked_rule_group)) == true_marks:
    #             if ret_rule_group is None:
    #                 ret_rule_group = [marked_rule_group]
    #             else:
    #                 for i in range(len(ret_rule_group)):
    #                     if getattr(TypeGroups, marked_rule_group) > getattr(TypeGroups, ret_rule_group[i]):
    #                         ret_rule_group[i] = marked_rule_group
    #                         added = True
    #                     if getattr(TypeGroups, marked_rule_group) < getattr(TypeGroups, ret_rule_group[i]):
    #                         added = True
    #                 if not added:
    #                     ret_rule_group.append(marked_rule_group)
    #
    #     if ret_rule_group is not None:
    #         ret_list = self.create_rule_group_class_list(ret_rule_group)
    #         return ret_list + self.unclassified_types
    #     else:
    #         if len(self.unclassified_types) == 0:
    #             return None
    #         else:
    #             return self.unclassified_types
    #
    # def is_type_in_rule_group(self, rule_group, type_):
    #     return type_ in getattr(TypeGroups, rule_group)

"""
TypeGroups composed by collections of types
"""
RealNumber = RuleGroupGenerator().create_rule_group_class("RealNumber")
Number = RuleGroupGenerator().create_rule_group_class("Number")
Integer = RuleGroupGenerator().create_rule_group_class("Integer")
Str = RuleGroupGenerator().create_rule_group_class("Str")
IterableDataStructure = RuleGroupGenerator().create_rule_group_class("IterableDataStructure")
IterableObject = RuleGroupGenerator().create_rule_group_class("IterableObject")
ByteSequence = RuleGroupGenerator().create_rule_group_class("ByteSequence")
