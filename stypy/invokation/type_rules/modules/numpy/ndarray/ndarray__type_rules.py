#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    '__getitem__': [
        ((Number,), DynamicType),
        ((slice,), DynamicType),
        ((IterableDataStructure,), DynamicType),
    ],

    '__setitem__': [
        ((Number, AnyType), DynamicType),
        ((slice, AnyType), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'argmin': [
        ((), DynamicType),
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
    ],

    'view': [
        ((), DynamicType),
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
    ],

    'reshape': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, VarArgs), DynamicType),
    ],

    'dot': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, DynamicType), DynamicType),
    ],

    'mean': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, VarArgs), DynamicType),
    ],

    'astype': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, VarArgs), DynamicType),
    ],

    'repeat': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
    ],

    'sum': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((dict,), DynamicType),
    ],

    'nonzero': [
        ((), DynamicType),
    ],

    '__div__': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        # ((Number, VarArgs), DynamicType),
        # ((IterableDataStructure, VarArgs), DynamicType),
    ],

    '__mod__': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        # ((Number, VarArgs), DynamicType),
        # ((IterableDataStructure, VarArgs), DynamicType),
    ],

    '__floordiv__': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        # ((Number, VarArgs), DynamicType),
        # ((IterableDataStructure, VarArgs), DynamicType),
    ],

    '__rdiv__': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        # ((Number, VarArgs), DynamicType),
        # ((IterableDataStructure, VarArgs), DynamicType),
    ],
}
