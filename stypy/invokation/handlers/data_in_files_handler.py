#!/usr/bin/env python
# -*- coding: utf-8 -*-
import imp
import inspect
import os
from abc import ABCMeta, abstractmethod

from abstract_call_handler import AbstractCallHandler
from stypy import stypy_parameters
from stypy.sgmc.sgmc_main import SGMC
from stypy.types.type_inspection import get_name, get_defining_module, get_defining_class, is_method
from stypy.types.type_intercession import get_superclasses


class DataInFilesHandler(AbstractCallHandler):
    """
    This call handler uses type rule files (Python files with a special structure) to determine acceptable parameters
    and return types for the calls of a certain module/class and its callable members. The handler dynamically search,
    load and use these rule files to resolve calls.
    """
    __metaclass__ = ABCMeta

    # Cache of found rule files
    type_rule_cache = dict()

    # Cache of not found rule files (to improve performance)
    unavailable_type_rule_cache = dict()

    def __init__(self, file_postfix, rule_getter, resolve_function_rules=False):
        self.unavailable_type_rule_cache = dict()
        self.type_rule_cache = dict()
        self.file_postfix = file_postfix
        self.rule_getter = rule_getter
        self.resolve_function_rules = resolve_function_rules

    def get_rule_file(self, parent_name, entity_name=None):
        """
        For a call to parent_name.entity_name(...), compose the name of the type rule file that will correspond to the
        entity or its parent, to look inside any of them for suitable rules to apply
        :param parent_name: Parent entity (module/class) name
        :param entity_name: Callable entity (function/method) name
        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)
        """
        if entity_name is None:
            return stypy_parameters.ROOT_PATH + stypy_parameters.RULE_FILE_PATH + parent_name.replace('.', '/') + "/" \
                   + parent_name.split('.')[-1] + self.file_postfix + ".py"

        return stypy_parameters.ROOT_PATH + stypy_parameters.RULE_FILE_PATH + parent_name.replace('.', '/') + "/" + \
               entity_name.split('.')[-1] + "/" + entity_name.split('.')[-1] + self.file_postfix + ".py"

    def __put_entity_into_unavailable_cache(self, entity):
        """
        Cache that holds those entities that have no rule file associated
        :param entity:
        :return:
        """
        try:
            self.unavailable_type_rule_cache[entity] = True
        except:
            pass

    def __put_entity_into_cache(self, entity, rules):
        """
        Cache that holds those entities that have a rule file associated
        :param entity:
        :return:
        """
        try:
            self.type_rule_cache[entity] = rules
        except:
            pass

    def __get_rules_from_unavailable_cache(self, entity):
        """
        Determines if there aren't rule files for an entity consulting the cache
        :param entity:
        :return:
        """
        try:
            return self.unavailable_type_rule_cache[entity]
        except:
            return False

    def __get_from_cache(self, entity):
        """
        Determines if there are rule files for an entity consulting the cache
        :param entity:
        :return:
        """
        try:
            return self.type_rule_cache[entity]
        except:
            return None

    def get_data_for_class(self, module, clazz, member=None):
        """
        Get the rules associated to an specific class of a certain module, optially providing a member to search rules
        :param module:
        :param clazz:
        :param member:
        :return:
        """
        # Rule file for this class
        rule_file = self.get_rule_file(module, get_name(clazz))

        # Not found a rule file: look over its superclasses
        if rule_file is None:
            # Are there superclasses?
            bases = get_superclasses(clazz)
            if len(bases) > 0:
                for bclass in bases:
                    if bclass is clazz:
                        continue
                    # Found suitable rules over any superclass?
                    rules = self.get_data_for_class(get_defining_module(bclass), bclass, member)
                    if rules is not None:
                        return rules

        if rule_file is None:
            return None

        # Do the rule file exit?
        file_exist = os.path.isfile(rule_file)
        if not file_exist:
            if 'stypy/sgmc/sgmc_cache/site_packages/' in rule_file:
                file_exist = os.path.isfile(rule_file.replace('stypy/sgmc/sgmc_cache/site_packages/', ''))
                if not file_exist:
                    return None
                else:
                    rule_file = rule_file.replace('stypy/sgmc/sgmc_cache/site_packages/', '')
            else:
                return None  # If not, we cannot handle this request

        file_ = rule_file.split('/')[-1][0:-3]

        try:
            module = imp.load_source(file_, rule_file)
            if member is None:
                entity_name = get_name(clazz)
            else:
                entity_name = get_name(member).split('.')[-1]

            # Is there a rule for the specific entity even if the container of the entity has a rule file?
            # This way rule files are used while they are created. All rule files declare a member called
            # type_rules_of_members
            rules = self.rule_getter(module, entity_name)

            if self.resolve_function_rules:
                if inspect.isfunction(rules):
                    rules = rules()

            # Cache loaded rules for the member
            if member is None:
                self.__put_entity_into_cache(clazz, rules)
            else:
                self.__put_entity_into_cache(member, rules)
            return rules
        except:
            bases = get_superclasses(clazz)
            if len(bases) > 0:
                for bclass in bases:
                    if bclass is clazz:
                        continue
                    # Found suitable rules over any superclass?
                    rules = self.get_data_for_class(get_defining_module(bclass), bclass, member)
                    if rules is not None:
                        return rules
            # Cache unexisting rules for the member
            # self.unavailable_type_rule_cache[entity_name] = True
            return None

    def get_data_for_module(self, module, member):
        """
        Get the rules associated for a certain member of a module
        :param module:
        :param member:
        :return:
        """
        if module is None:
            return None

        # Rule file for this module
        rule_file = self.get_rule_file(module)

        if rule_file is None:
            return None

        file_exist = os.path.isfile(rule_file)
        if not file_exist:
            rule_file = self.get_rule_file(SGMC.get_original_module_name(module))
            file_exist = os.path.isfile(rule_file)
            if not file_exist:
                return None

        file_ = rule_file.split('/')[-1][0:-3]

        try:
            module = imp.load_source(file_, rule_file)
            entity_name = get_name(member).split('.')[-1]

            # Is there a rule for the specific entity even if the container of the entity has a rule file?
            # This way rule files are used while they are created. All rule files declare a member called
            # type_rules_of_members
            rules = self.rule_getter(module, entity_name)

            if self.resolve_function_rules:
                if inspect.isfunction(rules):
                    rules = rules()

            # Cache loaded rules for the member
            if member is None:
                self.__put_entity_into_cache(module, rules)
            else:
                self.__put_entity_into_cache(member, rules)
            return rules
        except:
            # Cache unexisting rules for the member
            # self.unavailable_type_rule_cache[cache_name] = True
            return None

    def can_be_applicable_to(self, callable_entity):
        """
        This method determines if this call handler is able to respond to a call to callable_entity. The call handler
        respond to any callable code that has a rule file associated. This method search the rule file and, if found,
        loads and caches it for performance reasons. Cache also allows us to not to look for the same file on the
        hard disk over and over, saving much time. callable_entity rule files have priority over the rule files of
        their parent entity should both exist.

        :param callable_entity: Callable entity
        :return: bool
        """

        # No rule file for this callable (all options have been tried in a previous call)
        if self.__get_rules_from_unavailable_cache(callable_entity):
            return None

        """
        Rule file lookup order:

        Class:
        - Lookup class rule file
        - Lookup superclasses rule file
        - Lookup class module
        Class method:
        - Lookup class rule file
        - Lookup superclasses rule file
        Function:
        - Lookup module
        """
        # There are a rule file for this callable (from the cache)
        rules = self.__get_from_cache(callable_entity)
        if rules is not None:
            return rules

        # Obtain available rule files depending on the type of entity that is going to be called

        # Class method
        if is_method(callable_entity):
            class_ = get_defining_class(callable_entity)
            return self.get_data_for_class(get_defining_module(class_), class_, callable_entity)

        if inspect.isclass(callable_entity):
            rules = self.get_data_for_class(get_defining_module(callable_entity), callable_entity)
            if rules is None:
                rules = self.get_data_for_module(get_defining_module(callable_entity),
                                                 callable_entity)
            return rules

        # The only possibility left is that this is a function that belong to a module
        module = get_defining_module(callable_entity)
        if module is None and type(callable_entity).__name__ == "ufunc":
            module = "numpy"

        rules = self.get_data_for_module(module, callable_entity)
        if rules is None:
            pass
            # Cache unexisting rules for the member
            #self.__put_entity_into_unavailable_cache(callable_entity)

        return rules

    @abstractmethod
    def __call__(self, applicable_rules, localization, callable_, *arguments, **keyword_arguments):
        pass
