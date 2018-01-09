#!/usr/bin/env python
# -*- coding: utf-8 -*-
import core_language
import functions
import stypy_functions

"""
Helper functions to generate operator related nodes in the type inference AST
"""


# ##################################### OPERATORS #########################################


def create_binary_operator(op_name, op1, op2, line=0, column=0):
    """
    Creates an unary operator
    :param op_name:
    :param op1:
    :param op2:
    :param line:
    :param column:
    :return:
    """
    localization = stypy_functions.create_localization(line, column)

    binop_func = core_language.create_Name(stypy_functions.default_operator_call_name)
    binop = functions.create_call(binop_func, [localization, op_name, op1, op2])

    return binop


def create_unary_operator(op_name, op, line=0, column=0):
    """
    Creates a binary operator
    :param op_name:
    :param op:
    :param line:
    :param column:
    :return:
    """
    localization = stypy_functions.create_localization(line, column)

    unop_func = core_language.create_Name(stypy_functions.default_operator_call_name)
    unop = functions.create_call(unop_func, [localization, op_name, op])

    return unop
