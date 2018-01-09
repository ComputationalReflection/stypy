#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        ((Str, Integer,), types.NoneType),
    ],

    'warn': [
        ((Str,), types.NoneType),
        ((Str, type), types.NoneType),
        ((Str, type, Integer), types.NoneType),
    ],

    'resetwarnings': [
        ((), types.NoneType)
    ],

    'simplefilter': [
        ((Str,), types.NoneType),
        ((Str, type), types.NoneType),
        ((Str, type, Integer), types.NoneType),
        ((Str, type, Integer, bool), types.NoneType),
    ],

    'formatwarning': [
        ((Str, type, Str, Integer), types.NoneType),
        ((Str, type, Str, Integer, Integer), types.NoneType),
    ],

    'showwarning': [
        ((Str, type, Str, Integer), types.NoneType),
        ((Str, type, Str, Integer, file), types.NoneType),
        ((Str, type, Str, Integer, file, Integer), types.NoneType),
    ],

    'warnpy3k': [
        ((Str,), types.NoneType),
        ((Str, type), types.NoneType),
        ((Str, type, Integer), types.NoneType),
    ],

    'warn_explicit': [
        ((Str, type, Str, Integer), types.NoneType),
        ((Str, type, Str, Integer, types.ModuleType), types.NoneType),
        ((Str, type, Str, Integer, types.ModuleType, dict), types.NoneType),
        ((Str, type, Str, Integer, types.ModuleType, dict, dict), types.NoneType),
    ],
}
