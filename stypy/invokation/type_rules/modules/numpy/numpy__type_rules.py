#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyArrayOfFloat = numpy.array(0.0)
NumpyArrayOfFloatList = numpy.array([0.0])

type_rules_of_members = {
    'hypot': [
        ((Number, Number), numpy.float64()),
        ((Number, IterableDataStructure), NumpyArrayOfFloatList),
        ((numpy.ndarray, numpy.ndarray), TypeOfParam(1)),
        ((numpy.ndarray, Number), TypeOfParam(1)),
        ((IterableDataStructure, Number), NumpyArrayOfFloatList),
        ((Number, numpy.ndarray), TypeOfParam(2)),
    ],

    'dtype': [
        ((AnyType,), DynamicType),
        ((AnyType, bool), DynamicType),
        ((AnyType, bool, bool), DynamicType),
    ],

    'logical_not': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
    ],

    'logical_and': [
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
    ],

    'logical_or': [
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
    ],

    'logical_xor': [
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
    ],

    'negative': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
    ],

    'lookfor': [
        ((Str,), DynamicType),
        ((Str, VarArgs), DynamicType),
    ],

    'reciprocal': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, VarArgs), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'divide': [
        ((Number, Number), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((Number, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((Number, Number, VarArgs), DynamicType),
        ((IterableDataStructure, Number, VarArgs), DynamicType),
        ((Number, IterableDataStructure, VarArgs), DynamicType),
        ((IterableDataStructure, IterableDataStructure, VarArgs), DynamicType),
    ],

    'true_divide': [
        ((Number, Number), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((Number, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((Number, Number, VarArgs), DynamicType),
        ((IterableDataStructure, Number, VarArgs), DynamicType),
        ((Number, IterableDataStructure, VarArgs), DynamicType),
        ((IterableDataStructure, IterableDataStructure, VarArgs), DynamicType),
    ],

    'floor_divide': [
        ((Number, Number), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((Number, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((Number, Number, VarArgs), DynamicType),
        ((IterableDataStructure, Number, VarArgs), DynamicType),
        ((Number, IterableDataStructure, VarArgs), DynamicType),
        ((IterableDataStructure, IterableDataStructure, VarArgs), DynamicType),
    ],

    'fmod': [
        ((Number, Number), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((Number, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((Number, Number, VarArgs), DynamicType),
        ((IterableDataStructure, Number, VarArgs), DynamicType),
        ((Number, IterableDataStructure, VarArgs), DynamicType),
        ((IterableDataStructure, IterableDataStructure, VarArgs), DynamicType),
    ],

    'remainder': [
        ((Number, Number), DynamicType),
        ((IterableDataStructure, Number), DynamicType),
        ((Number, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((Number, Number, VarArgs), DynamicType),
        ((IterableDataStructure, Number, VarArgs), DynamicType),
        ((Number, IterableDataStructure, VarArgs), DynamicType),
        ((IterableDataStructure, IterableDataStructure, VarArgs), DynamicType),
    ],

    'log': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, VarArgs), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'log2': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, VarArgs), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],

    'log10': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, VarArgs), DynamicType),
        ((IterableDataStructure, VarArgs), DynamicType),
    ],
}
