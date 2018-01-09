import types

from stypy.errors import type_error

test_types = {
    'function_1': {
        'a': int,
        'x': type_error.StypyTypeError,
    },
    'function_2': {
        'a': int,
        'x': type_error.StypyTypeError,
    },
    '__main__': {
        'r1': int,
        'function_1': types.FunctionType,
        'r2': int,

        'function_2': types.FunctionType,
        'r3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
    },
}
