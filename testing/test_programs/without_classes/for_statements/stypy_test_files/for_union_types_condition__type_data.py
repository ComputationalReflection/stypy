from stypy.errors import type_error
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'c': union_type.UnionType.create_from_type_list([str, int, UndefinedType]),
        'r': union_type.UnionType.create_from_type_list([str, int, UndefinedType]),
        'x': union_type.UnionType.create_from_type_list([str, int, UndefinedType]),
    },
}
