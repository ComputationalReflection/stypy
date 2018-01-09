#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyArrayOfFloat = numpy.array(0.0)
NumpyArrayOfFloatList = numpy.array([0.0])

type_rules_of_members = {
    'random': [
        ((), numpy.float64),
        ((Integer,), DynamicType),
        ((IterableDataStructure,), DynamicType),
    ],

    'rand': [
        ((Integer,), DynamicType),
        ((Integer, VarArgs), DynamicType),
    ]
}
