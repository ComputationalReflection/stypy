import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'Foo': types.ClassType,
        'Foo2': types.ClassType,
        'r1': type_error.StypyTypeError,
        'x': int,
        'r2': type_error.StypyTypeError,
        'r3': slice,
    },
}
