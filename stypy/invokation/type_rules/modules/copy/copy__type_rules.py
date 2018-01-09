#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'copy': [
        ((AnyType,), TypeOfParam(1)),
    ],

    'deepcopy': [
        ((AnyType,), TypeOfParam(1)),
    ],
}
