#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyFloat = numpy.float64()


def same_rules_as(key):
    return type_rules_of_members[key]


type_rules_of_members = {
    '_wrapit': [
        ((AnyType, Str, AnyType, AnyType), DynamicType),
    ],

    'around': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'round': lambda: same_rules_as('around'),
    'round_': lambda: same_rules_as('around'),

    'clip': [
        ((Number, Number, Number), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'reshape': [
        ((Number,), DynamicType),
        ((Number, Number), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'sum': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'prod': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'cumsum': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'cumprod': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, dict), DynamicType),
    ],

    'argmin': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'all': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'any': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'argpartition': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'argsort': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'trace': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType, AnyType), DynamicType),
    ],

    'repeat': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'sort': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'nonzero': [
        ((IterableDataStructure,), DynamicType),
    ],

    'flatnonzero': [
        ((IterableDataStructure,), DynamicType),
    ],

    'count_nonzero': [
        ((IterableDataStructure,), int),
        ((IterableDataStructure, AnyType), DynamicType),
    ],
}
