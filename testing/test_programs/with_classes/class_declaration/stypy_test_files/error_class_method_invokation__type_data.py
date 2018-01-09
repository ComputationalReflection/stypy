import types
from stypy.errors import type_error
from stypy.types import union_type

u = union_type.UnionType(types.InstanceType, types.InstanceType)

test_types = {
    '__main__': {
        'FooParent': types.ClassType,
        'FooChild': types.ClassType,
        'o': u,
        'x': union_type.UnionType.create_from_type_list([str, int, list, bool]),
        'l': int,
        'o2': u,
        'r1': type_error.StypyTypeError,
        'r2': union_type.UnionType.create_from_type_list([str, int, list, bool]),
        'r3': type_error.StypyTypeError,
        'l2': int,
    },
}
