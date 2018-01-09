#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'pickle': [
        ((AnyType, types.FunctionType,), types.NoneType),
        ((AnyType, types.FunctionType, types.FunctionType), types.NoneType),
    ],
}
