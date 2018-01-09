import numpy

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'lookfor': [
        ((Str,), DynamicType),
        ((Str, VarArgs), DynamicType),
    ]
}