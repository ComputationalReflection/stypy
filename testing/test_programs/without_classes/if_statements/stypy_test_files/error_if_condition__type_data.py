import types
from stypy.errors.type_error import StypyTypeError
from stypy.types import union_type

test_types = {
    '__main__': {
        'x': StypyTypeError,
        'y': union_type.UnionType.create_from_type_list([str, bool]),
    },
}
