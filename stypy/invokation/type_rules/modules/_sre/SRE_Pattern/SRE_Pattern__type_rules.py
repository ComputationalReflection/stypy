#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

# https://docs.python.org/3/library/re.html#re-objects


type_rules_of_members = {
    'search': [
        ((Str,), DynamicType),
        ((Str, Integer), DynamicType),
        ((Str, Integer, Integer), DynamicType),
    ],

}
