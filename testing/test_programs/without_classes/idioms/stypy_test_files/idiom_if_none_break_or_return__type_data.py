import types

from stypy.errors.type_error import StypyTypeError
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'x': union_type.UnionType.create_from_type_list([int, types.NoneType]),
        'r': int,
    },
}
