#!/usr/bin/env python
# -*- coding: utf-8 -*-


from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {

    'unwrap': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((IterableDataStructure, Number, Integer), DynamicType),
    ],
    'interp': [
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType, AnyType, AnyType), DynamicType),
    ],

    'i0': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
    ],

    'sinc': [
        ((Number,), numpy.float64()),
        ((IterableDataStructure,), DynamicType),
    ],

    'gradient': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'diff': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'trapz': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'angle': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, bool), DynamicType),
    ],
}
