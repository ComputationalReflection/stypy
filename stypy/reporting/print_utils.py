#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from stypy.types.type_inspection import is_error
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.undefined_type import UndefinedType
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_function_ret_var_name

"""
Several functions to help printing elements with a readable style on error reports
"""

# These variables are used by stypy, therefore they should not be printed.
private_variable_names = ["__temp_tuple_assignment", "__temp_list_assignment",
                          "__temp_lambda", "__temp_call_assignment"]


def format_function_name(fname):
    """
    Prints a function name, considering lambda functions.
    :param fname: Function name
    :return: Proper function name
    """
    # if default_lambda_var_name in fname:
    #     return "lambda function"
    return fname


def is_private_variable_name(var_name):
    """
    Determines if a variable is a stypy private variable
    :param var_name: Variable name
    :return: bool
    """
    for private_name in private_variable_names:
        if private_name in var_name:
            return True

    return False


def get_type_str(type_):
    """
    Get the abbreviated str representation of a type for printing friendly error messages
    :param type_: Type
    :return: str
    """
    if is_error(type_):
        return "TypeError"

    return str(type_)


def get_param_position(source_code, param_number):
    """
    Get the offset of a parameter within a source code line that specify a method header. This is used to mark
    parameters with type errors when reporting them.

    :param source_code: Source code (method header)
    :param param_number: Number of parameter
    :return: Offset of the parameter in the source line
    """
    try:
        split_str = source_code.split(',')
        if param_number >= len(split_str):
            return 0

        if param_number == 0:
            name_and_first = split_str[0].split('(')
            offset = len(name_and_first[0]) + 1

            blank_offset = 0
            for car in name_and_first[1]:
                if car == " ":
                    blank_offset += 1
        else:
            offset = 0
            for i in xrange(param_number):
                offset += len(split_str[i]) + 1  # The comma also counts

            blank_offset = 0
            for car in split_str[param_number]:
                if car == " ":
                    blank_offset += 1

        return offset + blank_offset
    except:
        return -1


def print_type(obj):
    """
    Prints a type of an object
    :param obj:
    :return:
    """
    if obj is None:
        return "None"
    type_to_print = type(obj).__name__

    if isinstance(obj, TypeWrapper):
        type_to_print = str(obj)
    if is_error(obj):
        type_to_print = type(obj).__name__ + "(\"" + obj.message + "\")"

    if obj is UndefinedType:
        type_to_print = "UndefinedType"
    else:
        if type(obj) is types.TypeType:
            type_to_print = str(obj)

    return type_to_print


def print_context_contents(context):
    """
    Prints the contents of a context
    :param context:
    :return:
    """
    return_type = "NoneType"

    header = "\n[ ** Context: '" + context.context_name + "' ** ] \n"
    txt = "\tDeclared variables:\n"
    sorted_keys = sorted(context.types_of.keys())
    for name in sorted_keys:
        value = context.types_of[name]
        if name == default_function_ret_var_name:
            return_type = print_type(value)
            continue

        type_to_print = print_type(value)

        if "__stypy_auto_var" not in name:
            txt += "\t\t{0} = {1}\n".format(name, type_to_print)
    if len(context.globals) > 0:
        txt += "\tDeclared globals: ["
        for name in context.globals:
            txt += name + ", "
        txt = txt[:-2]
        txt += "]"

    return header + "Return type: " + return_type + "\n\n" + txt + "\n"
