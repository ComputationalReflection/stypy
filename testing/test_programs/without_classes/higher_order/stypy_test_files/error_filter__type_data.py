import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'l2': list,
        'f2': types.FunctionType,
        'other_l': list,
        'r1': int,
        'l3': list,
        'other_l2': list,
        'r2': type_error.StypyTypeError,
        'other_l3': type_error.StypyTypeError,
    },
}
