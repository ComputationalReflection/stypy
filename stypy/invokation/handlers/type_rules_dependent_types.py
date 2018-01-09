#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_groups import DependentType, DynamicType


def dependent_type_in_rule_params(params):
    """
    Check if a list of params has dependent types: Types that have to be called somewhat in order to obtain the
    real type they represent.
    :param params: List of types
    :return: bool
    """
    return len(filter(lambda par: isinstance(par, DependentType), params)) > 0


def get_arguments(argument_tuple, current_pos, rule_arity):
    """
    Obtain a list composed by the arguments present in argument_tuple, except the one in current_pos limited
    to rule_arity size. This is used when invoking dependent rules
    :param argument_tuple:
    :param current_pos:
    :param rule_arity:
    :return:
    """
    if rule_arity == 0:
        return []

    temp_list = []
    for i in xrange(len(argument_tuple)):
        if not i == current_pos:
            temp_list.append(argument_tuple[i])

    return tuple(temp_list[0:rule_arity])


def invoke_dependent_rules(localization, rule_params, arguments):
    """
        As we said, some rules may contain special types called DependentTypes. These types have to be invoked in
        order to check that the rule matches with the call or other necessary operations. Dependent types may have
        several forms, and are called with all the arguments that are checked against the type rule except the one
        that matches de dependent type, limited by the Dependent type declared rule arity. For example a Dependent
        Type may be defined like this (see type_groups.py for all the Dependent types defined):

        Overloads__eq__ = HasMember("__eq__", DynamicType, 1)

        This means that Overloads__eq__ matches with all the objects that has a method named __eq__ that has no
        predefined return type and an arity of 1 parameter. On the other hand, a type rule may be defined like this:

        ((Overloads__eq__, AnyType), DynamicType)

        This means that the type rule matches with a call that has a first argument which overloads the method
        __eq__ and any kind of second arguments. Although __eq__ is a method that should return bool (is the ==
        operator) this is not compulsory in Python, the __eq__ method may return anything and this anything will be
        the result of the rule. So we have to call __eq__ with the second argument (all the arguments but the one
        that matches with the DependentType limited to the declared dependent type arity), capture and return the
        result. This is basically the functionality of this method.

        Note that invocation to a method means that the type rule call handler (or another one) may be used again
        against the invoked method (__eq__ in our example).

        :param localization: Caller information
        :param rule_params: Rule file entry
        :param arguments: Arguments passed to the call that matches against the rule file.
        :return:
    """
    temp_rule = []
    for i in xrange(len(rule_params)):
        # Are we dealing with a dependent type?
        if isinstance(rule_params[i], DependentType):
            # Invoke it with the parameters we described previously
            correct_invokation, equivalent_type = rule_params[i](
                localization, *get_arguments(arguments, i, rule_params[i].call_arity))

            # Is the invocation correct?
            if not correct_invokation:
                # No, return that this rule do not really match
                return False, None, None
            else:
                # The equivalent type is the one determined by the dependent type rule invocation
                if equivalent_type is not None:
                    # By convention, if the declared rule result is DynamicType, it is substituted by its equivalent
                    # type. This is the most common case
                    if rule_params[i].expected_return_type is DynamicType:
                        return True, None, equivalent_type

                    # Some dependent types have a declared fixed return type (not like our previous example, which
                    # has DynamicType instead. In that case, if the dependent type invocation do not return the
                    # expected type, this means that the match is not valid and another rule has to be used to
                    # resolve the call.
                    if rule_params[i].expected_return_type is not type(equivalent_type):
                        return False, None, None
                else:
                    temp_rule.append(rule_params[i])
        else:
            temp_rule.append(rule_params[i])

    return True, tuple(temp_rule), None


def create_return_type(localization, ret_type, argument_types):
    """
    Create a suitable return type for the rule (if the return type is a dependent type, this invoked it against
    the call arguments to obtain it)
    :param localization: Caller information
    :param ret_type: Declared return type in a matched rule
    :param argument_types: Arguments of the call
    :return:
    """
    if isinstance(ret_type, DependentType):
        return ret_type(localization, argument_types)
    else:
        return ret_type
