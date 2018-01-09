import ast
import types

import stypy_functions_copy
import functions_copy
import core_language_copy

"""
Code that deals with various code idioms that can be optimized to better obtain the types of the variables used
on these idioms. The work with this file is unfinished, as not all the intended idioms are supported.

TODO: Finish this and its comments when idioms are fully implemented
"""

# Idiom constant names:

# Idiom identified, var of the type call, type
default_ret_tuple = False, None, None

may_be_type_func_name = "may_be_type"
may_not_be_type_func_name = "may_not_be_type"
may_be_var_name = "__may_be"
more_types_var_name = "__more_types_in_union"


def __has_call_to_type_builtin(test):
    if type(test) is ast.Call:
        if type(test.func) is ast.Name:
            if len(test.args) != 1:
                return False
            if test.func.id == "type":
                return True
    else:
        if hasattr(test, "left"):
            if type(test.left) is ast.Call:
                if len(test.comparators) != 1:
                    return False
                if type(test.left.func) is ast.Name:
                    if len(test.left.args) != 1:
                        return False
                    if test.left.func.id == "type":
                        return True
    return False


def __has_call_to_is(test):
    if len(test.ops) == 1:
        if type(test.ops[0]) is ast.Is:
            return True
    return False


def __is_type_name(test):
    if type(test) is ast.Name:
        name_id = test.id
        try:
            type_obj = eval(name_id)
            return type(type_obj) is types.TypeType
        except:
            return False
    return False


def type_is_idiom(test, visitor, context):
    """
    Idiom "type is"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.Compare:
        return default_ret_tuple

    if __has_call_to_type_builtin(test) and __has_call_to_is(test):
        if not (__has_call_to_type_builtin(test.comparators[0]) or __is_type_name(test.comparators[0])):
            return default_ret_tuple
        type_param = visitor.visit(test.left.args[0], context)
        if __is_type_name(test.comparators[0]):
            is_operator = visitor.visit(test.comparators[0], context)
        else:
            is_operator = visitor.visit(test.comparators[0].args[0], context)
            if not isinstance(is_operator[0], list):
                is_operator = ([is_operator[0]], is_operator[1])

        return True, type_param, is_operator

    return default_ret_tuple


def not_type_is_idiom(test, visitor, context):
    """
    Idiom "not type is"

    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.UnaryOp:
        return default_ret_tuple
    if type(test.op) is not ast.Not:
        return default_ret_tuple

    return type_is_idiom(test.operand, visitor, context)


def __get_idiom_type_param(test):
    return test.left.args[0]


def __set_type_implementation(if_test, type_, lineno, col_offset):
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        return stypy_functions_copy.create_set_type_of(param.id, type_, lineno, col_offset)
    if type(param) is ast.Attribute:
        obj_type, obj_var = stypy_functions_copy.create_get_type_of(param.value.id, lineno, col_offset)
        set_member = stypy_functions_copy.create_set_type_of_member(obj_var, param.attr, type_, lineno, col_offset)
        return stypy_functions_copy.flatten_lists(obj_type, set_member)

    return []


def __remove_type_from_union_implementation(if_test, type_, lineno, col_offset):
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        obj_type, obj_var = stypy_functions_copy.create_get_type_of(param.id, lineno, col_offset)
        remove_type_call = functions_copy.create_call(core_language_copy.create_Name("remove_type_from_union"),
                                                 [obj_var, type_], line=lineno, column=col_offset)
        set_type = stypy_functions_copy.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions_copy.flatten_lists(obj_type, set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions_copy.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions_copy.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions_copy.create_call(core_language_copy.create_Name("remove_type_from_union"),
                                                 [att_var, type_], line=lineno, column=col_offset)
        set_member = stypy_functions_copy.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions_copy.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


def set_type_of_idiom_var(idiom_name, if_branch, if_test, type_, lineno, col_offset):
    if idiom_name == "type_is":
        if if_branch == "if":
            return __set_type_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "else":
            return __remove_type_from_union_implementation(if_test, type_, lineno, col_offset)

    if idiom_name == "not_type_is":
        if_test = if_test.operand
        if if_branch == "if":
            return __remove_type_from_union_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "else":
            return __set_type_implementation(if_test, type_, lineno, col_offset)

    return []


# Recognized idioms
recognized_idioms = {
    "type_is": type_is_idiom,
    "not_type_is": not_type_is_idiom,
}

# Implementation of recognized idioms
recognized_idioms_functions = {
    "type_is": may_be_type_func_name,
    "not_type_is": may_not_be_type_func_name,
}


def get_recognized_idiom_function(idiom_name):
    """
    Gets the function that process an idiom once it has been recognized
    :param idiom_name: Idiom name
    :return:
    """
    return recognized_idioms_functions[idiom_name]


def is_recognized_idiom(test, visitor, context):
    """
    Check if the passed test can be considered an idioms

    :param test: Source code test
    :param visitor: Type inference visitor, to change generated instructions
    :param context: Context passed to the call
    :return: Tuple of values that identify if an idiom has been recognized and calculated data if it is been recognized
    """
    for idiom in recognized_idioms:
        result = recognized_idioms[idiom](test, visitor, context)
        if result[0]:
            temp_list = list(result)
            temp_list.append(idiom)
            return tuple(temp_list)

    return False, None, None, None
