from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'iteritems': [
        ((dict,), DynamicType),
        ((dict, dict), DynamicType),
    ],
}
