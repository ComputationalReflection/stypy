import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'Foo': types.ClassType,
        'f1': types.ClassType,
        'f2': types.ClassType,
        'r1': type_error.StypyTypeError,
        'r2': bool,
    },
}
