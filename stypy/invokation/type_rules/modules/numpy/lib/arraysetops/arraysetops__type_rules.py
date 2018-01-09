#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'ediff1d': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((IterableDataStructureWithTypedElements(Number),), DynamicType),
        ((numpy.ndarray,), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), AnyType), DynamicType),
        ((numpy.ndarray, AnyType), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), AnyType, AnyType), DynamicType),
        ((numpy.ndarray, AnyType, AnyType), DynamicType),
    ],
    'unique': [
        ((Number,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ]
}
