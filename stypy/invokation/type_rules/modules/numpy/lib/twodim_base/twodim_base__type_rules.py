#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'diag': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'triu': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'tril': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'diagflat': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],
}
