#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'filterwarnings': [
        ((Str,), types.NoneType),
        ((Str, dict), types.NoneType),
        ((Str, Str), types.NoneType),
        ((Str, Str, Str), types.NoneType),
        ((Str, Str, Str, Str), types.NoneType),
        ((Str, Str, Str, Str, Integer), types.NoneType),
        ((Str, Str, Str, Str, Integer, AnyType), types.NoneType),
        ((Str, Str, Str, Integer, AnyType), types.NoneType),
        ((Str, Str, Integer, AnyType), types.NoneType),
        ((Str, Integer, AnyType), types.NoneType),
        ((Str, Str, Str, Str, Integer,), types.NoneType),
        ((Str, Str, Str, Integer,), types.NoneType),
        ((Str, Str, Integer,), types.NoneType),
        ((Str, Integer, ), types.NoneType),
    ],

    'warn': [
        ((Str,), types.NoneType),
        ((Str, type), types.NoneType),
        ((Str, type, Integer), types.NoneType),
    ]
}
