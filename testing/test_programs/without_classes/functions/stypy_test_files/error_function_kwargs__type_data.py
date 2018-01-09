import types

from stypy.types import union_type
from stypy.errors import type_error

test_types = {
    'function': {
        'a': int,
        'kwargs': dict,
    },
    '__main__': {
        'function': types.FunctionType,

        'y': int,
        'y2': type_error.StypyTypeError,
    },
}
