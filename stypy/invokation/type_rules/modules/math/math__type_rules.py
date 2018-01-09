#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'pow': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat), float),
        ((CastsToFloat, RealNumber), float),
        ((CastsToFloat, CastsToFloat), float),
    ],

    'fsum': [
        ((IterableDataStructureWithTypedElements(RealNumber, CastsToFloat),), float),
    ],

    'cosh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'ldexp': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
    ],

    'hypot': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
        ((CastsToFloat, CastsToFloat,), float),
    ],

    'acosh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'tan': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'asin': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'isnan': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'log': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
        ((CastsToFloat, CastsToFloat,), float),
    ],

    'fabs': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'floor': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'atanh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'modf': [
        ((RealNumber,), tuple),
        ((CastsToFloat,), tuple),
    ],

    'sqrt': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'lgamma': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'frexp': [
        ((RealNumber,), tuple),
        ((CastsToFloat,), tuple),
    ],

    'degrees': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'log10': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'sin': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'asinh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'exp': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'atan': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'factorial': [
        ((RealNumber,), int),
        ((CastsToFloat,), int),
    ],

    'copysign': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
    ],

    'expm1': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'ceil': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'isinf': [
        ((RealNumber,), bool),
        ((CastsToFloat,), bool),
    ],

    'sinh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'trunc': [
        ((bool,), int),
        ((long,), long),
        ((int,), int),
        ((float,), int),
        ((Overloads__trunc__,), int),
    ],

    'cos': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'tanh': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'radians': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'atan2': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
    ],

    'erf': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'erfc': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'fmod': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, CastsToFloat,), float),
        ((CastsToFloat, RealNumber), float),
    ],

    'acos': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'log1p': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],

    'gamma': [
        ((RealNumber,), float),
        ((CastsToFloat,), float),
    ],
}
