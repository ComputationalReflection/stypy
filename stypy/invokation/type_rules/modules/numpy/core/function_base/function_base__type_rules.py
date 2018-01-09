# !/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.invokation.type_rules.type_groups.type_group_generator import *

NumpyFloat = numpy.float64()


def same_rules_as(key):
    return type_rules_of_members[key]


type_rules_of_members = {
    'linspace': [
        ((Number, Number), DynamicType),
        ((Number, Number, AnyType), DynamicType),
        ((Number, Number, AnyType, AnyType), DynamicType),
        ((Number, Number, AnyType, AnyType, AnyType), DynamicType),
        ((Number, Number, AnyType, AnyType, AnyType, AnyType), DynamicType),
    ],

}
