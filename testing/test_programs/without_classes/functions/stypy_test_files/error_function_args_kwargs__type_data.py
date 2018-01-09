import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'r1': str,
        'functionargs': types.FunctionType,
        'x1': type_error.StypyTypeError,
        'functionkw': types.FunctionType,
        'functionkw2': types.FunctionType,

        'r2': type_error.StypyTypeError,
        'x2': type_error.StypyTypeError,
        'r3': str,
        'x3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
    },
}
