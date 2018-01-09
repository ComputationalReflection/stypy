import types

from stypy.errors import type_error

test_types = {
    'function': {
    },
    '__main__': {
        'function': types.FunctionType,
        'y': int,
        'r1': int,
        'r2': type_error.StypyTypeError,
        'r3': type_error.StypyTypeError,
    },
}
