#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast

import data_structures

"""
Helper functions to create AST Nodes for basic Python language elements
"""

IF_NAME_MAIN_STYPY_FUNC = 'run_stypy_name_equals_main'
IF_NAME_NOT_MAIN_STYPY_FUNC = 'run_stypy_name_dont_equals_main'


# ############################################# BASIC LANGUAGE ELEMENTS ##############################################


def create_nested_attribute(owner_name, nested_attribute, context=ast.Load(), line=0, column=0):
    """
    Creates an attribute derived from the passed one, prepending the passed owner (i. e. turns (C, a.b.c) into C.a.b.c.
    This is used in class attribute desugaring
    :param owner_name: Class name
    :param nested_attribute: Attribute
    :param context:
    :param line:
    :param column:
    :return:
    """
    if isinstance(nested_attribute, ast.Name):
        attribute = ast.Attribute()
        attribute.attr = nested_attribute.id
        attribute.ctx = ast.Load()
        attribute.lineno = line
        attribute.col_offset = column
        attribute.value = create_Name(owner_name, True, line, column)

        return attribute
    else:
        attribute = ast.Attribute()
        attribute.attr = nested_attribute.attr
        attribute.ctx = ast.Load()
        attribute.lineno = line
        attribute.col_offset = column
        attribute.value = create_nested_attribute(owner_name, nested_attribute.value, context, line, column)
        return attribute


def create_attribute(owner_name, att_name, context=ast.Load(), line=0, column=0):
    """
    Creates an ast.Attribute node using the provided parameters to fill its field values

    :param owner_name: (str) owner name of the attribute (for instance, if the attribute is obj.method,
    the owner is "obj")
    :param att_name: (str) Name of the attribute ("method" in the previous example)
    :param context: ast.Load (default) or ast.Store)
    :param line: Line number (optional)
    :param column: Column offset (optional)
    :return: An AST Attribute node.
    """
    attribute = ast.Attribute()
    attribute.attr = att_name
    attribute.ctx = context
    attribute.lineno = line
    attribute.col_offset = column

    if isinstance(owner_name, str):
        attribute_name = ast.Name()
        attribute_name.ctx = ast.Load()
        attribute_name.id = owner_name
        attribute_name.lineno = line
        attribute_name.col_offset = column

        attribute.value = attribute_name
    else:
        attribute.value = owner_name

    return attribute


def create_Name(var_name, right_hand_side=True, line=0, column=0):
    """
    Creates an ast.Name node using the provided parameters to fill its field values

    :param var_name: (str) value of the name
    :param right_hand_side: ast.Load (default) or ast.Store
    :param line: Line number (optional)
    :param column: Column offset (optional)
    :return: An AST Name node.
    """
    name = ast.Name()
    name.id = var_name
    name.lineno = line
    name.col_offset = column

    if right_hand_side:
        name.ctx = ast.Load()
    else:
        name.ctx = ast.Store()

    return name


def create_NoneType():
    """
    Creates the types.NoneType name
    :return:
    """
    return create_Name("types.NoneType")


def create_None():
    """
    Creates the None name
    :return:
    """
    return create_Name("None")


def create_Assign(left_hand_side, right_hand_side):
    """
    Creates an Assign AST Node, with its left and right hand side
    :param left_hand_side: Left hand side of the assignment (AST Node)
    :param right_hand_side: Right hand side of the assignment (AST Node)
    :return: AST Assign node
    """
    right_hand_side.ctx = ast.Load()
    left_hand_side.ctx = ast.Store()
    return ast.Assign(targets=[left_hand_side], value=right_hand_side)


def create_str(s, line=0, col=0):
    """
    Creates an AST Str node with the passed contents
    :param s: Content of the AST Node
    :param line: Line
    :param col: Column
    :return: An AST Str
    """
    str_ = ast.Str()

    str_.s = s
    str_.lineno = line
    str_.col_offset = col

    return str_


def create_alias(name, alias="", asname=None):
    """
    Creates an AST Alias node

    :param name: Name of the aliased variable
    :param alias: Alias name
    :param asname: Name to put if the alias uses the "as" keyword
    :return: An AST alias node
    """
    alias_node = ast.alias()

    alias_node.alias = alias
    alias_node.asname = asname
    alias_node.name = name

    return alias_node


def create_importfrom(module, names, level=0, line=0, column=0):
    """
    Creates an AST ImportFrom node

    :param module: Module to import
    :param names: Members of the module to import
    :param level: Level of the import
    :param line: Line
    :param column: Column
    :return: An AST ImportFrom node
    """
    importfrom = ast.ImportFrom()
    importfrom.level = level
    importfrom.module = module

    if data_structures.is_iterable(names):
        importfrom.names = names
    else:
        importfrom.names = [names]

    importfrom.lineno = line
    importfrom.col_offset = column

    return importfrom


def create_num(n, lineno=0, col_offset=0):
    """
    Create an AST Num node

    :param n: Value
    :param lineno: line
    :param col_offset: column
    :return: An AST Num node
    """
    num = ast.Num()
    num.n = n
    num.lineno = lineno
    num.col_offset = col_offset

    return num


def create_bool(value, lineno=0, col_offset=0):
    """
    Create an bool value

    :param value: Value
    :param lineno: line
    :param col_offset: column
    :return: An AST Num node
    """
    num = ast.Name()
    num.id = str(value)
    num.lineno = lineno
    num.col_offset = col_offset

    return num


def create_type_tuple(*elems):
    """
    Creates an AST Tuple node

    :param elems: ELements of the tuple
    :return: AST Tuple node
    """
    tuple_ = ast.Tuple()

    tuple_.elts = []
    for elem in elems:
        tuple_.elts.append(elem)

    return tuple_


def is_main(node):
    """
    Determines if this node represent a main in Python (checks for the if __name__ == '__main__' statement)
    :param node:
    :return:
    """
    if type(node.test) is ast.Compare:
        if type(node.test.comparators) is list:
            if type(node.test.comparators[0]) is ast.Str:
                if node.test.comparators[0].s == '__main__':
                    if type(node.test.left) is ast.Name:
                        if node.test.left.id == '__name__':
                            return True
    return False
