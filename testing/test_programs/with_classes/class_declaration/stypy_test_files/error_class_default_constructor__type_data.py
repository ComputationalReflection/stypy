import types
from stypy.errors import type_error

test_types = {
    '__main__': {
        'Foo': types.ClassType,
        'f': type_error.StypyTypeError,
    },
}
