#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'tile': [
        ((IterableDataStructure, Integer), TypeOfParam(1)),
        ((IterableDataStructure, IterableDataStructureWithTypedElements(Integer)), TypeOfParam(1)),
    ],
}
