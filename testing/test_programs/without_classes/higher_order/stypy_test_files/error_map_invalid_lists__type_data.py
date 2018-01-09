from stypy.errors import type_error

test_types = {
    '__main__': {
        'l2': list,
        'l': list,
        'other_l2': list,
        'r1': type_error.StypyTypeError,

        'l3': list,
        'r2': type_error.StypyTypeError,
        'other_l3': type_error.StypyTypeError,

        'l4': list,
        'other_l4': type_error.StypyTypeError,
        'r3': type_error.StypyTypeError,
    },
}
