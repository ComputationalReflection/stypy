#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'einsum': [
        ((Number, VarArgs,), DynamicType),
        ((Number, VarArgs, dict), DynamicType),
        ((Str, VarArgs,), DynamicType),
        ((Str, VarArgs, dict), DynamicType),
        ((IterableDataStructure, VarArgs,), DynamicType),
        ((IterableDataStructure, VarArgs, dict), DynamicType)
    ],
}
