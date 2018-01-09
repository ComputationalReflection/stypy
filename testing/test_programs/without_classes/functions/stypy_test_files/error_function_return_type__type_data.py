import types

from stypy.errors import type_error

test_types = {
    'problematic_get': {
    },
    '__main__': {
        'problematic_get': types.FunctionType,
        'x': type_error.StypyTypeError,
    },
}
