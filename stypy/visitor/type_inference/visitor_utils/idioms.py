#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import types

import core_language
import functions
import stypy_functions

"""
Code that deals with various code idioms that can be optimized to better obtain the types of the variables used
on these idioms. The work with this file is unfinished, as not all the intended idioms are supported.
"""

# Idiom constant names:

# Idiom identified, var of the type call, type
default_ret_tuple = False, None, None

may_be_none_func_name = "may_be_none"
may_not_be_none_func_name = "may_not_be_none"

may_be_type_func_name = "may_be_type"
may_not_be_type_func_name = "may_not_be_type"
may_be_var_name = "__may_be"
more_types_var_name = "__more_types_in_union"

may_be_subtype_func_name = "may_be_subtype"
may_not_be_subtype_func_name = "may_not_be_subtype"
may_be_subtype_var_name = "__may_be_subtype"
more_subtypes_var_name = "__more_subtypes_in_union"

may_provide_member_func_name = "may_provide_member"
may_not_provide_member_func_name = "may_not_provide_member"
may_provide_var_name = "__may_provide_member"
more_types_provinding_var_name = "__more_types_with_member_in_union"

"""
Operations over the __dict__ member are excluded from idiom recognition code
"""
non_idiom_type_attributes = ['__dict__']


# ############################################### TYPE IS IDIOM ################################################


def type_of_non_writable_member(test):
    """
    Recognizes when an object is not writable in a type is idiom to avoid processing this line as an idiom. Currently
    just recognizes calls over the __dict__ attribute, but the feature is prepared to easily increase the number
    of attributes that are like this by just putting its name in the above list.
    :param test:
    :return:
    """
    global non_idiom_type_attributes
    if type(test) is ast.Compare:
        if hasattr(test, 'left'):
            if hasattr(test.left, 'args'):
                if len(test.left.args) > 0:
                    if isinstance(test.left.args[0], ast.Attribute):
                        if test.left.args[0].attr in non_idiom_type_attributes:
                            return True
    return False


def __has_call_to_type_builtin(test):
    """
    Recognizes type(...) calls
    :param test:
    :return:
    """
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
    """
    Recognizes is keyword usage
    :param test:
    :return:
    """
    if len(test.ops) == 1:
        if type(test.ops[0]) is ast.Is:
            return True
    return False


def is_type_name(test):
    """
    Recognizes type names
    :param test:
    :return:
    """
    # <type_name> constructs
    if type(test) is ast.Name:
        name_id = test.id
        try:
            type_obj = eval(name_id)
            return type(type_obj) is types.TypeType
        except:
            return False
    # type(obj) constructs
    if type(test) is ast.Call:
        if test.func.id == "type":
            return True

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
        if not (__has_call_to_type_builtin(test.comparators[0]) or is_type_name(test.comparators[0])):
            return default_ret_tuple
        if type_of_non_writable_member(test):
            return default_ret_tuple
        type_param = visitor.visit(test.left.args[0], context)
        if is_type_name(test.comparators[0]):
            is_operator = visitor.visit(test.comparators[0], context)
        else:
            is_operator = visitor.visit(test.comparators[0], context)
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
    """
    Extract the parameter of the type(...) call in the idiom
    :param test:
    :return:
    """
    if type(test) is ast.Call:
        if test.func.id == 'isinstance' or test.func.id == 'hasattr':
            return test.args[0]
    return test.left.args[0]


def __set_type_implementation(if_test, type_, lineno, col_offset):
    """
    Assigns the type of the condition when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    type_ = functions.create_call(type_, [], line=lineno, column=col_offset)
    if type(param) is ast.Name:
        return stypy_functions.create_set_type_of(param.id, type_, lineno, col_offset)
    if type(param) is ast.Attribute:
        obj_type, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, type_, lineno, col_offset)
        return stypy_functions.flatten_lists(obj_type, set_member)

    return []


def remove_type_from_union_implementation(if_test, type_, lineno, col_offset):
    return __remove_type_from_union_implementation(if_test, type_, lineno, col_offset)


def __remove_type_from_union_implementation(if_test, type_, lineno, col_offset):
    """
    Implements the call to remove_type_from_union when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        obj_type, obj_var = stypy_functions.create_get_type_of(param.id, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_type_from_union"),
                                                 [obj_var, type_], line=lineno, column=col_offset)
        set_type = stypy_functions.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions.flatten_lists(obj_type, set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_type_from_union"),
                                                 [att_var, type_], line=lineno, column=col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


# ############################################### IS NONE IDIOM ################################################


def __has_None_comparison(test):
    """
    Check if the expression "is None" appears in the code
    :param test:
    :return:
    """
    if len(test.comparators) == 1:
        if type(test.comparators[0]) is ast.Name:
            if test.comparators[0].id == "None":
                return True
    return False


def is_none(test, visitor, context):
    """
    Idiom "type is None"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.Compare:
        return default_ret_tuple

    if __has_call_to_is(test) and __has_None_comparison(test):
        type_param = visitor.visit(test.left, context)
        is_operator = visitor.visit(test.comparators[0], context)
        return True, type_param, is_operator

    return default_ret_tuple


def is_not_none(test, visitor, context):
    """
    Idiom "not type is None"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.UnaryOp:
        return default_ret_tuple
    if type(test.op) is not ast.Not:
        return default_ret_tuple

    return is_none(test.operand, visitor, context)


# ############################################### SUBTYPE IS IDIOM ################################################


def __remove_not_subtype_from_union_implementation(if_test, type_, lineno, col_offset):
    """
    Implements the call to remove_not_subtype_from_union when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        # obj_type, obj_var = stypy_functions.create_get_type_of(param.id, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_not_subtype_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_type = stypy_functions.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions.flatten_lists(set_type)  # obj_type, set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_not_subtype_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


def __remove_subtype_from_union_implementation(if_test, type_, lineno, col_offset):
    """
    Implements the call to remove_subtype_from_union when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        remove_type_call = functions.create_call(core_language.create_Name("remove_subtype_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_type = stypy_functions.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions.flatten_lists(set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_subtype_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


def __has_call_to_isinstance_builtin(test):
    """
    Recognizes calls to isinstance(...)
    :param test:
    :return:
    """
    if type(test) is ast.Call:
        if type(test.func) is ast.Name:
            if len(test.args) != 2:
                return False
            if test.func.id == "isinstance":
                return True

    return False


def is_instance_idiom(test, visitor, context):
    """
    Idiom "isinstance(type, obj)"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.Call:
        return default_ret_tuple

    if __has_call_to_isinstance_builtin(test):
        if is_type_name(test.args[1]):
            type_param = visitor.visit(test.args[1], context)
            is_instance_operand = visitor.visit(test.args[0], context)
            return True, type_param, is_instance_operand

    return default_ret_tuple


def not_is_instance_idiom(test, visitor, context):
    """
    Idiom "not isinstance(type, obj)"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.UnaryOp:
        return default_ret_tuple
    if type(test.op) is not ast.Not:
        return default_ret_tuple

    return is_instance_idiom(test.operand, visitor, context)


# ############################################### HASATTR IDIOM ################################################

def __has_call_to_hasattr_builtin(test):
    """
    Recognizes hasattr(...) calls
    :param test:
    :return:
    """
    if type(test) is ast.Call:
        if type(test.func) is ast.Name:
            if len(test.args) != 2:
                return False
            if test.func.id == "hasattr":
                return True

    return False


def __is_str_type(obj):
    """
    Recognizes strings in the code
    :param obj:
    :return:
    """
    return isinstance(obj, ast.Str)


def hasattr_idiom(test, visitor, context):
    """
    Idiom "hasattr(obj, name)"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.Call:
        return default_ret_tuple

    if __has_call_to_hasattr_builtin(test):
        if __is_str_type(test.args[1]):
            # stypy_functions.create_temp_Assign(, test.lineno, test.col_offset
            member_name = visitor.visit(test.args[1], context)
            hasattr_operand = visitor.visit(test.args[0], context)
            return True, member_name, hasattr_operand

    return default_ret_tuple


def not_hasattr_idiom(test, visitor, context):
    """
    Idiom "not hasattr(obj, name)"
    :param test:
    :param visitor:
    :param context:
    :return:
    """
    if type(test) is not ast.UnaryOp:
        return default_ret_tuple
    if type(test.op) is not ast.Not:
        return default_ret_tuple

    return hasattr_idiom(test.operand, visitor, context)


def __remove_not_member_provider_from_union_implementation(if_test, type_, lineno, col_offset):
    """
    Implements the call to remove_not_member_provider_from_union when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        remove_type_call = functions.create_call(core_language.create_Name("remove_not_member_provider_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_type = stypy_functions.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions.flatten_lists(set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_not_member_provider_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


def __remove_member_provider_from_union_implementation(if_test, type_, lineno, col_offset):
    """
    Implements the call to remove_member_provider_from_union when the idiom is recognized
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
    param = __get_idiom_type_param(if_test)
    if type(param) is ast.Name:
        remove_type_call = functions.create_call(core_language.create_Name("remove_member_provider_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_type = stypy_functions.create_set_type_of(param.id, remove_type_call, lineno, col_offset)

        return stypy_functions.flatten_lists(set_type)
    if type(param) is ast.Attribute:
        # Get the owner of the attribute
        obj_type_stmts, obj_var = stypy_functions.create_get_type_of(param.value.id, lineno, col_offset)
        # Get the current type of the owner of the attribute
        att_type_stmts, att_var = stypy_functions.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
        remove_type_call = functions.create_call(core_language.create_Name("remove_member_provider_from_union"),
                                                 [type_, if_test.args[1]], line=lineno, column=col_offset)
        set_member = stypy_functions.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
                                                               col_offset)
        return stypy_functions.flatten_lists(obj_type_stmts, att_type_stmts, set_member)

    return []


# ######################################### COMMON IDIOM FUNCTIONS #############################################


def set_type_of_idiom_var(idiom_name, if_branch, if_test, type_, lineno, col_offset):
    """
    Chooses the rigth implementation of type assignment of the idiom condition depending on the idiom type that has
    been recognized
    :param idiom_name:
    :param if_branch:
    :param if_test:
    :param type_:
    :param lineno:
    :param col_offset:
    :return:
    """
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

    if idiom_name == "subtype_is":
        if if_branch == "if":
            return __remove_not_subtype_from_union_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "else":
            return __remove_subtype_from_union_implementation(if_test, type_, lineno, col_offset)

    if idiom_name == "not_subtype_is":
        if_test = if_test.operand
        if if_branch == "else":
            return __remove_not_subtype_from_union_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "if":
            return __remove_subtype_from_union_implementation(if_test, type_, lineno, col_offset)

    if idiom_name == "may_provide_member":
        if if_branch == "if":
            return __remove_not_member_provider_from_union_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "else":
            return __remove_member_provider_from_union_implementation(if_test, type_, lineno, col_offset)

    if idiom_name == "may_not_provide_member":
        if_test = if_test.operand
        if if_branch == "else":
            return __remove_not_member_provider_from_union_implementation(if_test, type_, lineno, col_offset)
        if if_branch == "if":
            return __remove_member_provider_from_union_implementation(if_test, type_, lineno, col_offset)

    return []


# Recognized idioms
recognized_idioms = {
    "is_none": is_none,
    "is_not_none": is_not_none,
    "type_is": type_is_idiom,
    "not_type_is": not_type_is_idiom,
    "subtype_is": is_instance_idiom,
    "not_subtype_is": not_is_instance_idiom,
    "may_provide_member": hasattr_idiom,
    "may_not_provide_member": not_hasattr_idiom,
}

# Implementation of recognized idioms
recognized_idioms_functions = {
    "is_none": may_be_none_func_name,
    "is_not_none": may_not_be_none_func_name,
    "type_is": may_be_type_func_name,
    "not_type_is": may_not_be_type_func_name,
    "subtype_is": may_be_subtype_func_name,
    "not_subtype_is": may_not_be_subtype_func_name,
    "may_provide_member": may_provide_member_func_name,
    "may_not_provide_member": may_not_provide_member_func_name,
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
