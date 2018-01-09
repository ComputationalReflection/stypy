#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyArrayOfInt = numpy.array(0)
NumpyArrayOfIntArray = numpy.array([0])
NumpyArrayOfFloatArray = numpy.array([0.0])
NumpyFloat = numpy.float64()

type_rules_of_members = {
    'iscomplex': [
        ((Number,), bool),
        ((IterableDataStructure,), DynamicType),
    ],
    'nan_to_num': [
        ((RealNumber,), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((AnyType,), DynamicType),
    ],

    'real_if_close': [
        ((RealNumber,), DynamicType),
        ((RealNumber, RealNumber), DynamicType),
        ((IterableDataStructure,), DynamicType),
        ((IterableDataStructure, RealNumber), DynamicType),
    ],
}
