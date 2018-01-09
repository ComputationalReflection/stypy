#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    '_determine_error_states': [
        ((), types.NoneType),
    ],

    'det': [
        ((IterableDataStructure,), DynamicType),
    ],
}
