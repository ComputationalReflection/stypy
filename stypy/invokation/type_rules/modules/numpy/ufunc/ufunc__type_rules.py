#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *


type_rules_of_members = {
    'reduceat': [
        ((IterableDataStructure, IterableDataStructure), DynamicType),
        ((IterableDataStructure, IterableDataStructure, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, AnyType, AnyType), DynamicType),
        ((IterableDataStructure, IterableDataStructure, AnyType, AnyType, AnyType), DynamicType),
    ],
}
