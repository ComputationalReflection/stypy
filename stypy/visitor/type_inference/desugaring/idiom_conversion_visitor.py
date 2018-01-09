#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.type_inference_programs.aux_functions import *
from stypy.visitor.type_inference.visitor_utils.idioms import is_type_name

# ################################################### IDIOM VARIANT DEFINITIONS ######################################

default_ret_tuple = (False, "", None, None)


# ############### TYPE IS NONE ########################


def type_none_variant_type_equals(if_test):
    """
    Recognizes the if a == None idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "type_none"
    try:
        if type(if_test) is ast.Compare:
            if type(if_test.left) is ast.Name:
                if len(if_test.ops) == 1:
                    if type(if_test.ops[0]) is ast.Eq:
                        idiom_var = if_test.left
                        if len(if_test.comparators) == 1:
                            idiom_type = if_test.comparators[0]
                            if type(idiom_type) is ast.Name:
                                if idiom_type.id == "None":
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def not_type_none_variant_type_equals(if_test):
    """
    Recognizes the if not a == None idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_none"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if type(if_test.operand.left) is ast.Name:
                        if len(if_test.operand.ops) == 1:
                            if type(if_test.operand.ops[0]) is ast.Eq:
                                idiom_var = if_test.operand.left
                                if len(if_test.operand.comparators) == 1:
                                    idiom_type = if_test.operand.comparators[0]
                                    if type(idiom_type) is ast.Name:
                                        if idiom_type.id == "None":
                                            return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def type_is_not_none_variant_type_equals(if_test):
    """
    Recognizes the if a is not None idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_none"
    try:
        if type(if_test) is ast.Compare:
            if type(if_test.left) is ast.Name:
                if len(if_test.ops) == 1:
                    if type(if_test.ops[0]) is ast.IsNot:
                        idiom_var = if_test.left
                        if len(if_test.comparators) == 1:
                            idiom_type = if_test.comparators[0]
                            if type(idiom_type) is ast.Name:
                                if idiom_type.id == "None":
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


# ############### TYPE IS ########################


def type_is_variant_type_equals(if_test):
    """
    Recognizes the if type(a) == <type> idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "type_is"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.IsNot:
                    idiom_id = "not_type_is"
                if type(if_test.left) is ast.Call:
                    if if_test.left.func.id == 'type':
                        if len(if_test.left.args) == 1:
                            idiom_var = if_test.left.args[0]
                            if len(if_test.comparators) == 1:
                                idiom_type = if_test.comparators[0]
                                if is_type_name(idiom_type):
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def type_is_variant_class_is(if_test):
    """
    Recognizes the if a.__class__ == <type> idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "type_is"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.IsNot:
                    idiom_id = "not_type_is"
                if type(if_test.left) is ast.Attribute:
                    if if_test.left.attr == '__class__':
                        idiom_var = if_test.left.value
                        if len(if_test.comparators) == 1:
                            idiom_type = if_test.comparators[0]
                            if is_type_name(idiom_type):
                                return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def type_is_variant_equals_type(if_test):
    """
    Recognizes the if <type> == type(a) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "type_is"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.IsNot:
                    idiom_id = "not_type_is"
                if len(if_test.comparators) == 1:
                    if type(if_test.comparators[0]) is ast.Call:
                        if if_test.comparators[0].func.id == 'type':
                            idiom_var = if_test.comparators[0].args[0]
                            if type(if_test.left) is ast.Name:
                                idiom_type = if_test.left
                                if is_type_name(idiom_type):
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def type_is_variant_is_class(if_test):
    """
    Recognizes the if <type> == a.__class__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "type_is"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.IsNot:
                    idiom_id = "not_type_is"
                if len(if_test.comparators) == 1:
                    if type(if_test.comparators[0]) is ast.Attribute:
                        if if_test.comparators[0].attr == '__class__':
                            idiom_var = if_test.comparators[0].value
                            if type(if_test.left) is ast.Name:
                                idiom_type = if_test.left
                                if is_type_name(idiom_type):
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


# ################## NOT TYPE IS #################################


def not_type_is_variant_type_equals(if_test):
    """
    Recognizes the if not type(a) == <type> idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_is"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if type(if_test.operand.left) is ast.Call:
                        if if_test.operand.left.func.id == 'type':
                            if len(if_test.operand.left.args) == 1:
                                idiom_var = if_test.operand.left.args[0]
                                if len(if_test.operand.comparators) == 1:
                                    idiom_type = if_test.operand.comparators[0]
                                    if is_type_name(idiom_type):
                                        return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def not_type_is_variant_class_is(if_test):
    """
    Recognizes the if not a.__class__ == <type> idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_is"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if type(if_test.operand.left) is ast.Attribute:
                        if if_test.operand.left.attr == '__class__':
                            idiom_var = if_test.operand.left.value
                            if len(if_test.operand.comparators) == 1:
                                idiom_type = if_test.operand.comparators[0]
                                if is_type_name(idiom_type):
                                    return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def not_type_is_variant_equals_type(if_test):
    """
    Recognizes the if not <type> == type(a) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_is"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if len(if_test.operand.comparators) == 1:
                        if type(if_test.operand.comparators[0]) is ast.Call:
                            if if_test.operand.comparators[0].func.id == 'type':
                                idiom_var = if_test.operand.comparators[0].args[0]
                                if type(if_test.operand.left) is ast.Name:
                                    idiom_type = if_test.operand.left
                                    if is_type_name(idiom_type):
                                        return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


def not_type_is_variant_is_class(if_test):
    """
    Recognizes the if not <type> == a.__class__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_type_is"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if len(if_test.operand.comparators) == 1:
                        if type(if_test.operand.comparators[0]) is ast.Attribute:
                            if if_test.operand.comparators[0].attr == '__class__':
                                idiom_var = if_test.operand.comparators[0].value
                                if type(if_test.operand.left) is ast.Name:
                                    idiom_type = if_test.operand.left
                                    if is_type_name(idiom_type):
                                        return True, idiom_id, idiom_var, idiom_type
    except:
        return default_ret_tuple
    return default_ret_tuple


# ############### HASATTR ########################


def hasattr_variant_class_dict(if_test):
    """
    Recognizes the if <member name> in a.__class__.__dict__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "hasattr"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.In:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Attribute:
                            if if_test.comparators[0].attr == '__dict__':
                                if if_test.comparators[0].value.attr == '__class__':
                                    idiom_var = if_test.comparators[0].value.value
                                    if type(if_test.left) is ast.Str:
                                        member_name = if_test.left
                                        return True, idiom_id, idiom_var, member_name
    except:
        return default_ret_tuple
    return default_ret_tuple


def hasattr_variant_class_dict_has_key(if_test):
    """
    Recognizes the if type(a).__dict__.has_key(<member name>) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "hasattr"
    try:
        if type(if_test) is ast.Call:
            if if_test.func.attr == 'has_key':
                if len(if_test.args) == 1:
                    member_name = if_test.args[0]
                    if type(if_test.func.value) is ast.Attribute:
                        if if_test.func.value.attr == '__dict__':
                            if if_test.func.value.value.func.id == 'type':
                                if len(if_test.func.value.value.args) == 1:
                                    idiom_var = if_test.func.value.value.args[0]
                                    return True, idiom_id, idiom_var, member_name

    except:
        return default_ret_tuple
    return default_ret_tuple


def hasattr_variant_class_dict_in(if_test):
    """
    Recognizes the if <member_name> in type(a).__dict__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "hasattr"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.In:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Attribute:
                            if if_test.comparators[0].attr == '__dict__':
                                if type(if_test.comparators[0].value) is ast.Call:
                                    if type(if_test.comparators[0].value.func.id == 'type'):
                                        if len(if_test.comparators[0].value.args) == 1:
                                            idiom_var = if_test.comparators[0].value.args[0]
                                            if type(if_test.left) is ast.Str:
                                                member_name = if_test.left
                                                return True, idiom_id, idiom_var, member_name
    except:
        return default_ret_tuple
    return default_ret_tuple


def hasattr_variant_class_dir_in(if_test):
    """
    Recognizes the if <member_name> in dir(type(a)) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "hasattr"
    try:
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.In:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Call:
                            if if_test.comparators[0].func.id == 'dir':
                                if len(if_test.comparators[0].args) == 1:
                                    if type(if_test.comparators[0].args[0]) is ast.Call:
                                        if type(if_test.comparators[0].args[0].func.id == 'type'):
                                            if len(if_test.comparators[0].args[0].args) == 1:
                                                idiom_var = if_test.comparators[0].args[0].args[0]
                                                if type(if_test.left) is ast.Str:
                                                    member_name = if_test.left
                                                    return True, idiom_id, idiom_var, member_name
    except:
        return default_ret_tuple
    return default_ret_tuple


# ########################### NOT HASATTR #################################


def not_hasattr_variant_class_dict(if_test):
    """
    Recognizes the if not <member name> in a.__class__.__dict__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_hasattr"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if len(if_test.operand.ops) == 1:
                        if type(if_test.operand.ops[0]) is ast.In:
                            if len(if_test.operand.comparators) == 1:
                                if type(if_test.operand.comparators[0]) is ast.Attribute:
                                    if if_test.operand.comparators[0].attr == '__dict__':
                                        if if_test.operand.comparators[0].value.attr == '__class__':
                                            idiom_var = if_test.operand.comparators[0].value.value
                                            if type(if_test.operand.left) is ast.Str:
                                                member_name = if_test.operand.left
                                                return True, idiom_id, idiom_var, member_name

        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.NotIn:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Attribute:
                            if if_test.comparators[0].attr == '__dict__':
                                if if_test.comparators[0].value.attr == '__class__':
                                    idiom_var = if_test.comparators[0].value.value
                                    if type(if_test.left) is ast.Str:
                                        member_name = if_test.left
                                        return True, idiom_id, idiom_var, member_name
    except Exception as e:
        return default_ret_tuple
    return default_ret_tuple


def not_hasattr_variant_class_dict_has_key(if_test):
    """
    Recognizes the if not type(a).__dict__.has_key(<member name>) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_hasattr"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Call:
                    if if_test.operand.func.attr == 'has_key':
                        if len(if_test.operand.args) == 1:
                            member_name = if_test.operand.args[0]
                            if type(if_test.operand.func.value) is ast.Attribute:
                                if if_test.operand.func.value.attr == '__dict__':
                                    if if_test.operand.func.value.value.func.id == 'type':
                                        if len(if_test.operand.func.value.value.args) == 1:
                                            idiom_var = if_test.operand.func.value.value.args[0]
                                            return True, idiom_id, idiom_var, member_name

    except:
        return default_ret_tuple
    return default_ret_tuple


def not_hasattr_variant_class_dict_in(if_test):
    """
    Recognizes the if not <member_name> in type(a).__dict__ idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_hasattr"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if len(if_test.operand.ops) == 1:
                        if type(if_test.operand.ops[0]) is ast.In:
                            if len(if_test.operand.comparators) == 1:
                                if type(if_test.operand.comparators[0]) is ast.Attribute:
                                    if if_test.operand.comparators[0].attr == '__dict__':
                                        if type(if_test.operand.comparators[0].value) is ast.Call:
                                            if type(if_test.operand.comparators[0].value.func.id == 'type'):
                                                if len(if_test.operand.comparators[0].value.args) == 1:
                                                    idiom_var = if_test.operand.comparators[0].value.args[0]
                                                    if type(if_test.operand.left) is ast.Str:
                                                        member_name = if_test.operand.left
                                                        return True, idiom_id, idiom_var, member_name
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.NotIn:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Attribute:
                            if if_test.comparators[0].attr == '__dict__':
                                if type(if_test.comparators[0].value) is ast.Call:
                                    if type(if_test.comparators[0].value.func.id == 'type'):
                                        if len(if_test.comparators[0].value.args) == 1:
                                            idiom_var = if_test.comparators[0].value.args[0]
                                            if type(if_test.left) is ast.Str:
                                                member_name = if_test.left
                                                return True, idiom_id, idiom_var, member_name
    except:
        return default_ret_tuple
    return default_ret_tuple


def not_hasattr_variant_class_dir_in(if_test):
    """
    Recognizes the if not <member_name> in dir(type(a)) idiom
    :param if_test:
    :return:
    """
    global default_ret_tuple

    idiom_id = "not_hasattr"
    try:
        if type(if_test) is ast.UnaryOp:
            if type(if_test.op) is ast.Not:
                if type(if_test.operand) is ast.Compare:
                    if len(if_test.operand.ops) == 1:
                        if type(if_test.operand.ops[0]) is ast.In:
                            if len(if_test.operand.comparators) == 1:
                                if type(if_test.operand.comparators[0]) is ast.Call:
                                    if if_test.operand.comparators[0].func.id == 'dir':
                                        if len(if_test.operand.comparators[0].args) == 1:
                                            if type(if_test.operand.comparators[0].args[0]) is ast.Call:
                                                if type(if_test.operand.comparators[0].args[0].func.id == 'type'):
                                                    if len(if_test.operand.comparators[0].args[0].args) == 1:
                                                        idiom_var = if_test.operand.comparators[0].args[0].args[0]
                                                        if type(if_test.operand.left) is ast.Str:
                                                            member_name = if_test.operand.left
                                                            return True, idiom_id, idiom_var, member_name
        if type(if_test) is ast.Compare:
            if len(if_test.ops) == 1:
                if type(if_test.ops[0]) is ast.NotIn:
                    if len(if_test.comparators) == 1:
                        if type(if_test.comparators[0]) is ast.Call:
                            if if_test.comparators[0].func.id == 'dir':
                                if len(if_test.comparators[0].args) == 1:
                                    if type(if_test.comparators[0].args[0]) is ast.Call:
                                        if type(if_test.comparators[0].args[0].func.id == 'type'):
                                            if len(if_test.comparators[0].args[0].args) == 1:
                                                idiom_var = if_test.comparators[0].args[0].args[0]
                                                if type(if_test.left) is ast.Str:
                                                    member_name = if_test.left
                                                    return True, idiom_id, idiom_var, member_name
    except:
        return default_ret_tuple
    return default_ret_tuple


# ############## VARIANT HANDLER TABLE ##########################

"""
Recognized idiom variants list
"""
recognized_idiom_variants = [
    type_none_variant_type_equals,
    not_type_none_variant_type_equals,
    type_is_not_none_variant_type_equals,
    type_is_variant_type_equals,
    type_is_variant_class_is,
    type_is_variant_equals_type,
    type_is_variant_is_class,
    not_type_is_variant_type_equals,
    not_type_is_variant_class_is,
    not_type_is_variant_equals_type,
    not_type_is_variant_is_class,
    hasattr_variant_class_dict,
    hasattr_variant_class_dict_has_key,
    hasattr_variant_class_dict_in,
    hasattr_variant_class_dir_in,
    not_hasattr_variant_class_dict,
    not_hasattr_variant_class_dict_has_key,
    not_hasattr_variant_class_dict_in,
    not_hasattr_variant_class_dir_in,
]


# #################################### IDIOM VARIANTS CANONICAL FORM DEFINITIONS ########################

def create_type_none_canonical_idiom(old_node, idiom_var, idiom_type):
    """
    Code of the canonical form of the type none idiom (obj is None)
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param idiom_type: Type of the idiom
    :return: New code for the if node
    """
    compare = ast.Compare()

    compare.col_offset = old_node.col_offset
    compare.lineno = old_node.lineno
    compare.comparators = list()
    compare.comparators.append(idiom_type)
    compare.left = idiom_var
    compare.left.col_offset = old_node.col_offset
    compare.left.lineno = old_node.lineno
    compare.left.ctx = ast.Load()
    compare.ops = list()
    compare.ops.append(ast.Is())

    return compare


def create_not_type_none_canonical_idiom(old_node, idiom_var, idiom_type):
    """
    Code of the canonical form of the not type none idiom (not obj is None)
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param idiom_type: Type of the idiom
    :return: New code for the if node
    """
    compare = ast.Compare()

    compare.col_offset = old_node.col_offset
    compare.lineno = old_node.lineno
    compare.comparators = list()
    compare.comparators.append(idiom_type)
    compare.left = idiom_var
    compare.left.col_offset = old_node.col_offset
    compare.left.lineno = old_node.lineno
    compare.left.ctx = ast.Load()
    compare.ops = list()
    compare.ops.append(ast.Is())

    not_ = ast.UnaryOp()
    not_.operand = compare
    not_.col_offset = compare.col_offset
    not_.lineno = compare.lineno
    not_.op = ast.Not()

    return not_


def create_type_is_canonical_idiom(old_node, idiom_var, idiom_type):
    """
    Code of the canonical form of the type is idiom (type(obj) is type)
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param idiom_type: Type of the idiom
    :return: New code for the if node
    """
    compare = ast.Compare()

    compare.col_offset = old_node.col_offset
    compare.lineno = old_node.lineno
    compare.comparators = list()
    compare.comparators.append(idiom_type)
    compare.left = ast.Call()
    compare.left.args = list()
    compare.left.args.append(idiom_var)
    compare.left.col_offset = old_node.col_offset
    compare.left.func = ast.Name()
    compare.left.func.id = 'type'
    compare.left.func.ctx = ast.Load()
    compare.left.func.col_offset = old_node.col_offset
    compare.left.func.lineno = old_node.lineno
    compare.left.keywords = []
    compare.left.starargs = None
    compare.left.kwargs = None
    compare.ops = list()
    compare.ops.append(ast.Is())

    return compare


def create_not_type_is_canonical_idiom(old_node, idiom_var, idiom_type):
    """
    Code of the canonical form of the not type is idiom (not type(obj) is type)
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param idiom_type: Type of the idiom
    :return: New code for the if node
    """
    compare = ast.Compare()

    compare.col_offset = old_node.col_offset
    compare.lineno = old_node.lineno
    compare.comparators = list()
    compare.comparators.append(idiom_type)
    compare.left = ast.Call()
    compare.left.args = list()
    compare.left.args.append(idiom_var)
    compare.left.col_offset = old_node.col_offset
    compare.left.func = ast.Name()
    compare.left.func.id = 'type'
    compare.left.func.ctx = ast.Load()
    compare.left.func.col_offset = old_node.col_offset
    compare.left.func.lineno = old_node.lineno
    compare.left.keywords = []
    compare.left.starargs = None
    compare.left.kwargs = None
    compare.ops = list()
    compare.ops.append(ast.Is())

    not_ = ast.UnaryOp()
    not_.operand = compare
    not_.col_offset = compare.col_offset
    not_.lineno = compare.lineno
    not_.op = ast.Not()

    return not_


def create_hasattr_canonical_idiom(old_node, idiom_var, attr_name):
    """
    Code of the canonical form of the hasattr idiom (hasattr(obj, 'attr'))
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param attr_name: Attribute used in the idiom
    :return: New code for the if node
    """
    call = ast.Call()

    call.args = list()
    call.args.append(idiom_var)
    call.args.append(attr_name)
    call.col_offset = old_node.col_offset
    call.lineno = old_node.lineno
    call.func = ast.Name()
    call.func.id = 'hasattr'
    call.func.ctx = ast.Load()
    call.func.col_offset = old_node.col_offset
    call.func.lineno = old_node.lineno
    call.keywords = []
    call.starargs = None
    call.kwargs = None

    return call


def create_not_hasattr_canonical_idiom(old_node, idiom_var, attr_name):
    """
    Code of the canonical form of the not hasattr idiom (not hasattr(obj, 'attr'))
    :param old_node: Current if code
    :param idiom_var: Variable used in the idiom
    :param attr_name: Attribute used in the idiom
    :return: New code for the if node
    """
    call = ast.Call()

    call.args = list()
    call.args.append(idiom_var)
    call.args.append(attr_name)
    call.col_offset = old_node.col_offset
    call.lineno = old_node.lineno
    call.func = ast.Name()
    call.func.id = 'hasattr'
    call.func.ctx = ast.Load()
    call.func.col_offset = old_node.col_offset
    call.func.lineno = old_node.lineno
    call.keywords = []
    call.starargs = None
    call.kwargs = None

    not_ = ast.UnaryOp()
    not_.operand = call
    not_.col_offset = call.col_offset
    not_.lineno = call.lineno
    not_.op = ast.Not()

    return not_


"""
Functions that implement the canonical forms of each idioms.
"""
recognized_idiom_canonical_forms = {
    "type_none": create_type_none_canonical_idiom,
    "not_type_none": create_not_type_none_canonical_idiom,
    "type_is": create_type_is_canonical_idiom,
    "not_type_is": create_not_type_is_canonical_idiom,
    "hasattr": create_hasattr_canonical_idiom,
    "not_hasattr": create_not_hasattr_canonical_idiom,
}


class IdiomConversionVisitor(ast.NodeTransformer):
    """
    This transformer ensures that all the idiom variants that are recognized by stypy are transformed to its base
    equivalent form, in order to process them accordingly without spawning repeated code for each recognized code
    pattern. This means that we have set a canonical form for each idiom and all the possible variants are transformed
    to this canonical form prior to perform the type inference process. This way, only one type of processing per idiom
    is needed
    """

    def __visit_instruction_body(self, body):
        """
        Visit the if body, applying the idiom on Ifs that are inside If bodies.
        :param body:
        :return:
        """
        new_stmts = []

        for stmt in body:
            stmts = self.visit(stmt)
            if isinstance(stmts, list):
                new_stmts.extend(stmts)
            else:
                new_stmts.append(stmts)

        return new_stmts

    def visit_If(self, node):
        """
        Check if conditions to search for recognized idiom patterns.
        :param node:
        :return:
        """
        for func in recognized_idiom_variants:
            ret_tuple = func(node.test)
            if ret_tuple[0]:
                node.test = recognized_idiom_canonical_forms[ret_tuple[1]](node, ret_tuple[2], ret_tuple[3])
                break

        node.body = self.__visit_instruction_body(node.body)
        node.orelse = self.__visit_instruction_body(node.orelse)
        return node
