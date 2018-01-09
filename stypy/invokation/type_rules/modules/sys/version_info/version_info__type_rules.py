from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    '__getitem__': [
        ((Integer,), int),
        ((slice,), DynamicType),
    ],
}