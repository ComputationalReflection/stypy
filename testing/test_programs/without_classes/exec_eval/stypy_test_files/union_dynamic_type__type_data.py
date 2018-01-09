import types
from stypy.types import union_type
from stypy.errors.type_error import StypyTypeError
from stypy.invokation.type_rules.type_groups.type_group_generator import DynamicType

test_types = {
    '__main__': {
        'r': union_type.UnionType.create_from_type_list([str, DynamicType]),
        'x': DynamicType,
        'y': DynamicType,
        'y2': union_type.UnionType.create_from_type_list([bool, DynamicType]),
    },
}
