from stypy.types import union_type
from stypy.errors import type_error

test_types = {
    '__main__': {
        'x': union_type.UnionType.create_from_type_list([float, bool]),
        'c': type_error.StypyTypeError,
    }, 
}
