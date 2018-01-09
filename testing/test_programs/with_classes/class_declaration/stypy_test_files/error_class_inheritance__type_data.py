import types
from stypy.errors import type_error

test_types = {
    '__main__': {
        'exceptions': types.ModuleType,
        'MiException': types.ClassType,
        'MiExceptionWrong': type_error.StypyTypeError,
    },
}
