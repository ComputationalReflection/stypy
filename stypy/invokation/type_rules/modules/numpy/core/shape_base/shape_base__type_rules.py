#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyFloat = numpy.float64()


def same_rules_as(key):
    return type_rules_of_members[key]


type_rules_of_members = {
    'atleast_2d': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'hstack': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'vstack': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'dstack': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'block': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'stack': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'concatenate': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'hsplit': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],

    'split': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],

    'vsplit': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],

    'dsplit': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],

    'array_split': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), DynamicType),
    ],
}
