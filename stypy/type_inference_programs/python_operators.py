#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains functions that will be called from type inference code to deal with Python operator calls. It also
contains the logic to convert from operator names to its symbolic representation and viceversa.
"""

# Table to transform operator symbols to operator names (the name of the function that have to be called to implement
# the operator functionality in the builtin operator type inference module
__symbol_to_operator_table = {
    '+': 'add',
    '&': 'and_',
    'in': 'contains',
    '==': 'eq',
    '//': 'floordiv',
    '>=': 'ge',
    '[]': 'getitem',
    '>': 'gt',
    '+=': 'iadd',
    '&=': 'iand',
    '//=': 'ifloordiv',
    '<<=': 'ilshift',
    '%=': 'imod',
    '*=': 'imul',
    '~': 'inv',
    '|=': 'ior',
    '**=': 'ipow',
    '>>=': 'irshift',
    'is': 'is_',
    'is not': 'is_not',
    '-=': 'isub',
    '/=': 'itruediv',
    '^=': 'ixor',
    '<=': 'le',
    '<<': 'lshift',
    '>>': 'rshift',
    '<': 'lt',
    '%': 'mod',
    '*': 'mul',
    '!=': 'ne',
    '-': 'sub',  # beware of neg (unary)
    '/': 'truediv',
    '^': 'xor',
    '|': 'or_',
    'mult': 'mul',
    'and': 'and_keyword',
    'not': 'not_',
    'or': 'or_keyword',
    'div': 'div',
    'div=': 'div',
    'isnot': 'is_not',
    '**': 'pow',
    'notin': 'contains',
    'uadd': '__pos__',
    'usub': '__neg__',
}

# Table to perform the opposite operation than the previous one
__operator_to_symbol_table = {
    'or_keyword': '|',
    'and_keyword': '&',
    'lte': '<=',
    'gte': '>=',
    'eq': '==',
    'is_': 'is',
    'ior': '|=',
    'iand': '&=',
    'getitem': '[]',
    'imod': '%=',
    'not_': 'not',
    'xor': '^',
    'contains': 'in',
    'ifloordiv': '//=',
    'noteq': '!=',
    'is_not': 'isnot',
    'floordiv': '//',
    'mod': '%',
    'ixor': '^=',
    'ilshift': '<<=',
    'and_': '&',
    'add': '+',
    'mul': '*',
    'mult': '*',
    'sub': '-',
    'itruediv': '/=',
    'truediv': '/',
    'div': 'div',  # Integer division. We cannot use / as it is the float division (truediv)
    'lt': '<',
    'irshift': '>>=',
    'isub': '-=',
    'inv': '~',
    'lshift': '<<',
    'rshift': '>>',
    'iadd': '+=',
    'gt': '>',
    'pow': '**',
    'bitor': '|',
    'bitand': '&',
    'bitxor': '^',
    'invert': '~',
    '__pos__': 'uadd',
    '__neg__': 'usub',
}


# ###################################### OPERATOR REPRESENTATION CONVERSION #######################################

def operator_name_to_symbol(operator_name):
    """
    Transform an operator name to its symbolic representation (example: 'add' -> '+'. If no symbol is available, return
    the passed name
    :param operator_name: Operator name
    :return: Operator symbol
    """
    try:
        return __operator_to_symbol_table[operator_name]
    except KeyError:
        return operator_name


def operator_symbol_to_name(operator_symbol):
    """
    Transform an operator symbol to its function name (example: '+' -> 'add'.
    :param operator_symbol: Operator symbol
    :return: Operator name
    """
    return __symbol_to_operator_table[operator_symbol]
