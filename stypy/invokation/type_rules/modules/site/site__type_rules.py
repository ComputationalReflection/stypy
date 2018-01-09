#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.undefined_type import UndefinedType

type_rules_of_members = {
    'Quitter': [
        ((), UndefinedType),
        ((AnyType,), UndefinedType),
    ],

    '_Printer': [
        ((), types.NoneType),
    ],

    '_Helper': [
        ((), str),
        ((AnyType,), str),
    ],
}
