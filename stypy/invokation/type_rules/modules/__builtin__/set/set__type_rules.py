#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    'set': [
        ((IterableDataStructure,), DynamicType),
    ],
    'difference': [
        ((set,), DynamicType),
    ],
    'add': [
        ((AnyType, ), DynamicType),
    ],

    '__getitem__': [
        ((Integer,), DynamicType),
        ((slice,), DynamicType),
    ],

    'pop': [
        ((), DynamicType),
    ],
}
