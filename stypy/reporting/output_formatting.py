#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group import BaseTypeGroup
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_lambda_var_name
from stypy import types


def format_function_name(fname):
    """
    Prints a function name, considering lambda functions.
    :param fname: Function name
    :return: Proper function name
    """
    if default_lambda_var_name in fname:
        return "lambda function"
    return fname


def get_name(obj):
    """
    Gets the name of a Python object
    :param obj:
    :return:
    """
    if isinstance(obj, BaseTypeGroup):
        return str(obj)

    if hasattr(obj, "__name__"):
        return obj.__name__

    if hasattr(type(obj), "__name__"):
        return type(obj).__name__

    return str(obj)


def format_arguments(name, rules, arguments, call_arity):
    """
    Pretty-print error message when no type rule for the member matches with the arguments of the call
    :param name: Member name
    :param rules: Rules tied to this member name
    :param arguments: Call arguments
    :param call_arity: Call arity
    :return:
    """
    params_strs = [""] * call_arity
    first_rule = True
    arities = []

    # Problem with argument number?
    rules_with_enough_arguments = False
    for (params_in_rules, return_type) in rules:
        rule_len = len(params_in_rules)
        if rule_len not in arities:
            arities.append(rule_len)

        if len(params_in_rules) == call_arity:
            rules_with_enough_arguments = True
            for i in xrange(call_arity):
                value = get_name(params_in_rules[i])
                if value not in params_strs[i]:
                    if not first_rule:
                        params_strs[i] += " U "
                    params_strs[i] += value

            first_rule = False

    if not rules_with_enough_arguments:
        str_arities = ""
        for i in xrange(len(arities)):
            str_arities += str(arities[i])
            if len(arities) > 1:
                if i == (len(arities) - 2):
                    str_arities += " or "
                else:
                    if i != (len(arities) - 1):
                        str_arities += ", "

        return "The invocation was performed with {0} argument(s), but only {1} argument(s) are accepted".format(
            call_arity,
            str_arities)

    repr_ = ""
    for str_ in params_strs:
        repr_ += str_ + ", "

    return name + "(" + repr_[:-2] + ") expected"


# ########################################## PRETTY-PRINTING FUNCTION CALLS ##########################################


def __type_error_str(arg):
    """
    Helper function of the following one.
    If arg is a type error, this avoids printing all the TypeError information and only prints the name. This is
    convenient when pretty-printing calls and its passed parameters to report errors, because if we print the full
    error information (the same one that is returned by stypy at the end) the message will be unclear.
    :param arg:
    :return:
    """
    if isinstance(arg, TypeError):
        return "TypeError"
    else:
        if isinstance(arg, types.union_type.UnionType):
            return str(arg)
        if isinstance(arg, types.standard_wrapper.StandardWrapper):
            return str(arg)
        if arg is types.undefined_type.UndefinedType:
            return "UndefinedType"
        try:
            return type(arg).__name__
        except:
            return str(arg)


def __format_type_list(*arg_types, **kwargs_types):
    """
    Pretty-print passed parameter list
    :param arg_types:
    :param kwargs_types:
    :return:
    """
    arg_str_list = map(lambda elem: __type_error_str(elem), arg_types[0])
    arg_str = ""
    for arg in arg_str_list:
        arg_str += arg + ", "

    if len(arg_str) > 0:
        arg_str = arg_str[:-2]

    kwarg_str_list = map(lambda elem: __type_error_str(elem), kwargs_types)
    kwarg_str = ""
    for arg in kwarg_str_list:
        kwarg_str += arg + ", "

    if len(kwarg_str) > 0:
        kwarg_str = kwarg_str[:-1]
        kwarg_str = '{' + kwarg_str + '}'

    return arg_str, kwarg_str


def __format_callable(callable_):
    """
    Pretty-print a callable entity
    :param callable_:
    :return:
    """
    if hasattr(callable_, "__name__"):
        return callable_.__name__
    else:
        return str(callable_)


def format_call(callable_, arg_types, kwarg_types):
    """
    Pretty-print calls and its passed parameters, for error reporting, using the previously defined functions
    :param callable_:
    :param arg_types:
    :param kwarg_types:
    :return:
    """
    arg_str, kwarg_str = __format_type_list(arg_types, kwarg_types.values())
    callable_str = __format_callable(callable_)
    if len(kwarg_str) == 0:
        return "\t" + callable_str + "(" + arg_str + ")"
    else:
        return "\t" + callable_str + "(" + arg_str + ", " + kwarg_str + ")"
