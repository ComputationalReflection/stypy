#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'allclose': [
        ((Number, Number), bool()),
        ((Number, Number, AnyType), bool()),
        ((Number, Number, AnyType, AnyType), bool()),
        ((Number, Number, AnyType, AnyType, AnyType), bool()),
        ((IterableDataStructure, IterableDataStructure), bool()),
        ((IterableDataStructure, IterableDataStructure, AnyType), bool()),
        ((IterableDataStructure, IterableDataStructure, AnyType, AnyType), bool()),
        ((IterableDataStructure, IterableDataStructure, AnyType, AnyType, AnyType), bool()),
        ((IterableDataStructure, Number), bool()),
        ((IterableDataStructure, Number, AnyType), bool()),
        ((IterableDataStructure, Number, AnyType, AnyType), bool()),
        ((IterableDataStructure, Number, AnyType, AnyType, AnyType), bool()),
        ((Number, IterableDataStructure), bool()),
        ((Number, IterableDataStructure, AnyType), bool()),
        ((Number, IterableDataStructure, AnyType, AnyType), bool()),
        ((Number, IterableDataStructure, AnyType, AnyType, AnyType), bool()),
    ],

    'array_repr': [
        ((numpy.ndarray,), str),
        ((numpy.ndarray, None), str),
        ((numpy.ndarray, None, None), str),
        ((numpy.ndarray, None, None, None), str),
        ((numpy.ndarray, Integer), str),
        ((numpy.ndarray, Integer, Integer), Integer),
        ((numpy.ndarray, Integer, Integer, Integer), str),
    ],

    'geterr': [
        ((), DynamicType)
    ],
    'seterr': [
        ((), types.NoneType),
        ((str,), types.NoneType),
        ((str, str), types.NoneType),
        ((str, str, str), types.NoneType),
        ((str, str, str, str), types.NoneType),
        ((str, str, str, str, str), types.NoneType),
        ((dict,), types.NoneType),
    ],
    'convolve': [
        ((Number, Number,), DynamicType),
        ((Number, Number, AnyType,), DynamicType),
        ((IterableDataStructure, IterableDataStructure,), DynamicType),
        ((IterableDataStructure, IterableDataStructure, AnyType,), DynamicType),
        ((IterableDataStructure, Number,), DynamicType),
        ((IterableDataStructure, Number, AnyType,), DynamicType),
        ((Number, IterableDataStructure,), DynamicType),
        ((Number, IterableDataStructure, AnyType,), DynamicType),
    ],

    'cross': [
        (
            (IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number),),
            DynamicType),
        ((numpy.ndarray, numpy.ndarray,), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType,
          AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType, AnyType), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType,
          AnyType, AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType,
          AnyType, AnyType, AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType, AnyType, AnyType, AnyType), DynamicType),
    ],

    'ones': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],

    'ascontiguousarray': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
    ],

    'outer': [
        (
            (IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number),),
            DynamicType),
        ((numpy.ndarray, numpy.ndarray,), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType), DynamicType),

    ],

    'tensordot': [
        (
            (IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number),),
            DynamicType),
        ((numpy.ndarray, numpy.ndarray,), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), IterableDataStructureWithTypedElements(Number), AnyType),
         DynamicType),
        ((numpy.ndarray, numpy.ndarray, AnyType), DynamicType),

    ],

    'zeros_like': [
        ((Number,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((Number, AnyType), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((Number, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
        ((Number, AnyType, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],

    'roll': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],
    'rollaxis': [
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, AnyType, AnyType), DynamicType),
    ],
}
