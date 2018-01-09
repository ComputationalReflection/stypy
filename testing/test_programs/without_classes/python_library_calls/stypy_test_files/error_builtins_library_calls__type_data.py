import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'Foo': types.ClassType,
        'err': type_error.StypyTypeError,
        'r': int,
        'r2': int,
        'r3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
        'r5': int,
        'err2': type_error.StypyTypeError,
        'r6': type_error.StypyTypeError,
        'r7': bool,
        'err3': type_error.StypyTypeError,
        'r8': type_error.StypyTypeError,
        'err4': type_error.StypyTypeError,
        'words': list,
        'S': list,
        'err5': type_error.StypyTypeError,
        'r9': float,
        'err6': type_error.StypyTypeError,
    },
}
