from stypy.errors import type_error

test_types = {
    '__main__': {
        'words': list,
        'normal_list': list,
        'r1': type_error.StypyTypeError,
        'r2': type_error.StypyTypeError,
        'r3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
        'r5': float,
    },
}
