from stypy.errors import type_error
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'elem': type_error.StypyTypeError,
        'r': UndefinedType,
    },
}
