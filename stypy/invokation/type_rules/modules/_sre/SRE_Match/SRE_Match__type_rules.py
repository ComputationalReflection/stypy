#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

# https://docs.python.org/2.0/lib/match-objects.html


type_rules_of_members = {
    'groups': [
        ((), DynamicType),
        ((Integer,), DynamicType),
    ],

}
