#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'uniform': [
        ((), numpy.float64()),
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
    ],
    'random': [
        ((), numpy.float64),
        ((Integer,), DynamicType),
        ((IterableDataStructure,), DynamicType),
    ],
    'randint': [
        ((Integer,), numpy.int32),
        ((Integer, AnyType), DynamicType),
        ((Integer, AnyType, AnyType), DynamicType),
        ((Integer, AnyType, AnyType, AnyType), DynamicType),
    ],

    'rand': [
        ((Integer,), DynamicType),
        ((Integer, VarArgs), DynamicType),
    ]
}