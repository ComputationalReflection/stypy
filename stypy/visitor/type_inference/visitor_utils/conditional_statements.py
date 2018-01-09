#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast

import data_structures

"""
Helper functions to create conditional statements
"""


# ############################################### IF STATEMENTS ######################################################


def create_if(test, body, orelse=list()):
    """
    Creates an If AST Node, with its body and else statements
    :param test: Test of the if statement
    :param body: Statements of the body part
    :param orelse: Statements of the else part (optional
    :return: AST If Node
    """
    if_ = ast.If()

    if_.test = test
    if data_structures.is_iterable(body):
        if_.body = body
    else:
        if_.body = [body]

    if data_structures.is_iterable(orelse):
        if_.orelse = orelse
    else:
        if_.orelse = [orelse]

    return if_
