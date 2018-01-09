import types

from stypy.errors import type_error

test_types = {
    'function_1': {
        'x': str,
    },

    'function_2': {
        'x': list,
    },

    'function_3': {
        'x': int,
    },
    '__main__': {
        'function_1': types.FunctionType,
        'function_2': types.FunctionType,
        'function_3': types.FunctionType,

        'r1': type_error.StypyTypeError,
        'r2': type_error.StypyTypeError,
        'r3': int,
    },
}
