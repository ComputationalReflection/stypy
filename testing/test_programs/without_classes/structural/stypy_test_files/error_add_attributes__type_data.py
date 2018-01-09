import types
from stypy.errors import type_error
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type

test_types = {
    '__main__': {
        'func': types.LambdaType,
        'class_func': types.LambdaType,
        'f': types.InstanceType,
        'f2': types.InstanceType,
        'r1': int,
        'r2': type_error.StypyTypeError,
        'r3': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r4': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'r5': str
    },
}
