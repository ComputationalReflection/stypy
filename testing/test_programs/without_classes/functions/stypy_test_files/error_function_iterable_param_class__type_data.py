import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'fun1': types.FunctionType,

        'r1': type_error.StypyTypeError,
        'r2': type_error.StypyTypeError,
    },
}
