#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    '__getslice__': [
        ((Integer, Integer), DynamicType),
        ((Overloads__trunc__, Integer), DynamicType),
        ((Integer, Overloads__trunc__), DynamicType),
        ((Overloads__trunc__, Overloads__trunc__), DynamicType),
        ((Integer, Integer), DynamicType),
        ((Overloads__trunc__, Integer), DynamicType),
        ((Integer, Overloads__trunc__), DynamicType),
        ((Overloads__trunc__, Overloads__trunc__), DynamicType),
    ],

    '__rmul__': [
        ((Integer,), tuple)
    ],

    '__lt__': [
        ((AnyType,), types.NotImplementedType)
    ],

    'tuple': [
        ((), tuple),
        ((Str,), tuple),
        ((IterableObject,), tuple),
    ],

    'index': [
        ((AnyType,), int),
        ((AnyType, Integer), int),
    ],

    '__new__': [
        ((SubtypeOf(tuple)), tuple),
        ((SubtypeOf(tuple), VarArgType), tuple),
    ],

    '__contains__': [
        ((AnyType,), bool),
    ],

    '__len__': [
        ((), int),
    ],

    '__ne__': [
        ((AnyType,), types.NotImplementedType)
    ],

    '__getitem__': [
        ((Integer,), DynamicType),
        ((slice,), DynamicType),
    ],

    # insert can be invoked with the following number of parameters: [3]
    'insert': [
        ((Integer, AnyType), types.NoneType),
        ((Overloads__trunc__, AnyType), types.NoneType),
    ],

    '__iter__': [
        ((), ExtraTypeDefinitions.tupleiterator),
    ],

    '__gt__': [
        ((AnyType,), types.NotImplementedType)
    ],

    '__eq__': [
        ((AnyType,), types.NotImplementedType)
    ],

    'count': [
        ((AnyType,), int)
    ],

    '__add__': [
        ((tuple,), tuple),
    ],

    '__le__': [
        ((AnyType,), types.NotImplementedType)
    ],

    '__mul__': [
        ((Integer,), tuple),
    ],

    '__ge__': [
        ((AnyType,), types.NotImplementedType)
    ],

    '__getnewargs__': [
        ((), tuple),
    ],
}
