#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    # '__getslice__': [
    #     ((Integer, Integer), DynamicType),
    #     ((Overloads__trunc__, Integer), DynamicType),
    #     ((Integer, Overloads__trunc__), DynamicType),
    #     ((Overloads__trunc__, Overloads__trunc__), DynamicType),
    #     ((Integer, Integer), DynamicType),
    #     ((Overloads__trunc__, Integer), DynamicType),
    #     ((Integer, Overloads__trunc__), DynamicType),
    #     ((Overloads__trunc__, Overloads__trunc__), DynamicType),
    # ],

    'append': [
        ((AnyType,), types.NoneType)
    ],

    'pop': [
        ((), DynamicType),
    ],

    'popleft': [
        ((), DynamicType),
    ],
    #
    # 'remove': [
    #     ((AnyType,), types.NoneType)
    # ],
    #
    # '__rmul__': [
    #     ((Integer,), list)
    # ],
    #
    # '__lt__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
    #
    # 'extend': [
    #     ((IterableObject,), types.NoneType)
    # ],

    'deque': [
        ((), list),
        ((Str,), list),
        ((IterableObject,), list),
    ],

    # 'index': [
    #     ((AnyType,), int),
    #     ((AnyType, Integer), int),
    # ],
    #
    # '__delslice__': [
    #     ((Integer, Integer), types.NoneType),
    #     ((Overloads__trunc__, Integer), types.NoneType),
    #     ((Integer, Overloads__trunc__), types.NoneType),
    #     ((Overloads__trunc__, Overloads__trunc__), types.NoneType),
    #     ((Integer, Integer), types.NoneType),
    #     ((Overloads__trunc__, Integer), types.NoneType),
    #     ((Integer, Overloads__trunc__), types.NoneType),
    #     ((Overloads__trunc__, Overloads__trunc__), types.NoneType),
    # ],
    #
    # '__new__': [
    #     ((SubtypeOf(list)), list),
    #     ((SubtypeOf(list), VarArgType), list),
    #     # (type, VarArgType): first_param_is_a_subtype_of('list', list), TODO
    # ],
    #
    # '__contains__': [
    #     ((AnyType,), bool),
    # ],
    #
    # '__len__': [
    #     ((), int),
    # ],
    #
    # 'sort': [
    #     ((), types.NoneType),
    #     ((Has__call__,), types.NoneType),
    #     ((Has__call__, Has__call__), types.NoneType),
    #     ((Has__call__, Has__getitem__), types.NoneType),
    # ],
    #
    # '__ne__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
    #
    # '__getitem__': [
    #     ((Integer,), DynamicType),
    #     ((slice,), DynamicType),
    # ],
    #
    # # insert can be invoked with the following number of parameters: [3]
    # 'insert': [
    #     ((Integer, AnyType), types.NoneType),
    #     ((Overloads__trunc__, AnyType), types.NoneType),
    # ],
    #
    # '__iter__': [
    #     ((), ExtraTypeDefinitions.listiterator),
    # ],
    #
    # '__gt__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
    #
    # '__eq__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
    #
    'reverse': [
        ((), types.NoneType),
    ],
    #
    'count': [
        ((AnyType,), int)
    ],
    #
    # '__delitem__': [
    #     ((slice,), types.NoneType),
    #     ((Integer,), types.NoneType),
    # ],
    #
    # '__reversed__': [
    #     ((), ExtraTypeDefinitions.listreverseiterator),
    # ],
    #
    # '__imul__': [
    #     ((Integer,), DynamicType),
    # ],
    #
    # '__setslice__': [
    #     ((Integer, Integer, AnyType), types.NoneType),
    # ],
    #
    # '__setitem__': [
    #     ((Integer, AnyType), types.NoneType),
    #     ((slice, IterableDataStructure), types.NoneType),
    #     ((CastsToIndex, AnyType), types.NoneType),
    # ],
    #
    # '__add__': [
    #     ((list,), DynamicType),
    # ],
    #
    # '__iadd__': [
    #     ((list,), DynamicType),
    # ],
    #
    # '__le__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
    #
    # '__mul__': [
    #     ((Integer,), DynamicType),
    # ],
    #
    # '__ge__': [
    #     ((AnyType,), types.NotImplementedType)
    # ],
}
