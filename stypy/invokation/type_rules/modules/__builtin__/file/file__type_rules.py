#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    'read': [
        ((), str),
        ((Integer,), str),
    ],

    'write': [
        ((Str,), types.NoneType),
    ],
}
