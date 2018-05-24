#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'load_source': [
        ((Str,), DynamicType),
        ((Str, Str), DynamicType),
        ((Str, Str, Str), DynamicType),
    ],
}