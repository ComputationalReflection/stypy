import types

from stypy.types import union_type
from stypy.errors import type_error

test_types = {
    '__main__': {
        'math': types.ModuleType,
        'get_str': types.FunctionType,
        'r1': type_error.StypyTypeError,
        'r2': union_type.UnionType.create_from_type_list([int, str]),
        'r3': float,
    },
}
