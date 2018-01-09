#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'array': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType, AnyType), DynamicType),
    ],
    'empty': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],
    'arange': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
    ],
    'bincount': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],
    'einsum': [
        ((Number, VarArgs,), DynamicType),
        ((Number, VarArgs, dict), DynamicType),
        ((Str, VarArgs,), DynamicType),
        ((Str, VarArgs, dict), DynamicType),
        ((IterableDataStructure, VarArgs,), DynamicType),
        ((IterableDataStructure, VarArgs, dict), DynamicType)
    ],

    'dot': [
        ((Number, Number,), DynamicType),
        ((IterableDataStructure, IterableDataStructure,), DynamicType),
        ((Number, Number, DynamicType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, DynamicType), DynamicType),
    ],

    'inner': [
        ((Number, Number,), DynamicType),
        ((IterableDataStructure, IterableDataStructure,), DynamicType),
        ((Number, Number, DynamicType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, DynamicType), DynamicType),
    ],

    'zeros': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'unpackbits': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'set_typeDict': [
        ((dict, ), types.NoneType),
    ],

    'set_string_function': [
        ((types.FunctionType,), types.NoneType),
        ((types.FunctionType, int), types.NoneType),
    ]
}
