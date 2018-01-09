import types

from stypy.errors import type_error

test_types = {
    'comparations': {
        'a': int,
        'b': int,
        'c': int,
        'Foo': types.ClassType,
        'c0': type_error.StypyTypeError,
        'c1': type_error.StypyTypeError
    },
    '__main__': {
    },
}
