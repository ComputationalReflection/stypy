import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'WrongFoo': types.ClassType,
        'x': type_error.StypyTypeError,
    },
}
