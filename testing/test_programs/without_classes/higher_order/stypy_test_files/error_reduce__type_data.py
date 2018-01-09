from stypy.errors import type_error

test_types = {
    '__main__': {
        'l': list,
        'other_l': type_error.StypyTypeError,
        'r1': type_error.StypyTypeError,

        'l3': list,
        'r2': type_error.StypyTypeError,
        'other_l2': type_error.StypyTypeError,

        'other_l3': int,
        'r3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
        'other_l4': type_error.StypyTypeError,
    },
}
