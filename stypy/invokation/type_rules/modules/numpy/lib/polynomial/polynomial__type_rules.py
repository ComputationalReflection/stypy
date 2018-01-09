#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'poly1d': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, bool), DynamicType),
        ((IterableDataStructure, bool, Str), DynamicType),
        ((Integer, ), DynamicType),
    ],

    'polyfit': [
        ((IterableDataStructure, IterableDataStructure, Integer), DynamicType),
        ((IterableDataStructure, IterableDataStructure, Integer, DynamicType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, Integer, DynamicType, DynamicType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, Integer, DynamicType, DynamicType, DynamicType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, Integer, DynamicType, DynamicType, DynamicType, DynamicType), DynamicType),
    ]
}