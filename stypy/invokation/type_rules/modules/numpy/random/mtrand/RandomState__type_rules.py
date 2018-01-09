#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyArrayOfFloat = numpy.array(0.0)
NumpyArrayOfFloatList = numpy.array([0.0])

type_rules_of_members = {
    # 'random_sample': [
    #     ((), float),
    #     ((Number,), DynamicType),
    #     ((Number, VarArgs), DynamicType),
    # ],
}