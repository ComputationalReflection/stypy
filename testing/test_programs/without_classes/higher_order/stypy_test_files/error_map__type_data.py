import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'str_l': list,
        'l': list,
        'other_l': list,
        'r1': type_error.StypyTypeError,
        'f': types.FunctionType,
        'other_l2': list,
        'r2': type_error.StypyTypeError,
        'f2': types.FunctionType,
        'other_l3': list,
        'x': int,
        'other_l4': type_error.StypyTypeError,
    },
}
