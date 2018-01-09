import types
from stypy.errors import type_error

test_types = {
    'alias': {
        'r': type_error.StypyTypeError
    },

    '__main__': {
        'cos': list,
        'aliased': list,
        'alias': types.FunctionType,
    },
}
