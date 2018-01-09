import types

from stypy.errors import type_error
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type

test_types = {
    '__main__': {
        'r': union_type.UnionType.create_from_type_list([types.NoneType, int, UndefinedType]),
        'r2': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'g': types.FunctionType,
        'f': types.FunctionType,
    },
}
