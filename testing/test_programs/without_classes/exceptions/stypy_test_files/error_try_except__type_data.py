import types

from stypy.errors import type_error
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'math': types.ModuleType,
        'a': union_type.UnionType.create_from_type_list([int, str, UndefinedType]),
        'r1': str,
        'r2': type_error.StypyTypeError,
    },
}
