#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'ctime': [
        ((), str),
        ((RealNumber,), str),
        ((types.NoneType,), str),
        ((CastsToFloat,), str),
    ],
    'clock': [
        ((), float),
    ],
    'time': [
        ((), float),
    ],
    'strptime': [
        ((Str, Str), time.struct_time),
        ((bytearray,), time.struct_time),
        ((Str,), time.struct_time),
        ((bytearray, Str), time.struct_time),
    ],
    'gmtime': [
        ((), time.struct_time),
        ((RealNumber,), time.struct_time),
        ((types.NoneType,), time.struct_time),
        ((CastsToFloat,), time.struct_time),
    ],
    'mktime': [
        ((time.struct_time,), float),
    ],
    'sleep': [
        ((RealNumber,), types.NoneType),
        ((CastsToFloat,), types.NoneType),
    ],
    'asctime': [
        ((), str),
        ((time.struct_time,), str),
    ],
    'strftime': [
        ((Str,), str),
        ((Str, time.struct_time), str),
    ],
    'localtime': [
        ((), time.struct_time),
        ((RealNumber,), time.struct_time),
        ((types.NoneType,), time.struct_time),
        ((CastsToFloat,), time.struct_time),
    ],
}
