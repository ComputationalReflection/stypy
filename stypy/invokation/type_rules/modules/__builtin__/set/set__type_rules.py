#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    'update': [
        ((IterableDataStructure,), DynamicType),
    ],
    'intersection_update': [
        ((IterableDataStructure,), DynamicType),
    ],
    'difference_update': [
        ((IterableDataStructure,), DynamicType),
    ],
    'symmetric_difference_update': [
        ((IterableDataStructure,), DynamicType),
    ],
    'remove': [
        ((IterableDataStructure,), DynamicType),
    ],
    'discard': [
        ((IterableDataStructure,), DynamicType),
    ],
    'clear': [
        ((), DynamicType),
    ],
    'set': [
        ((IterableDataStructure,), DynamicType),
        ((), set),
    ],
    'difference': [
        ((set,), DynamicType),
    ],
    'add': [
        ((AnyType,), DynamicType),
    ],

    '__getitem__': [
        ((Integer,), DynamicType),
        ((slice,), DynamicType),
    ],

    'pop': [
        ((), DynamicType),
    ],
}
