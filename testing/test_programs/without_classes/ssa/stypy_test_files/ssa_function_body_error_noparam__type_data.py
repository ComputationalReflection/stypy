import types

from stypy.errors import type_error
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'r': UndefinedType,
        'r2': UndefinedType,
        'g': types.FunctionType,
        'f': types.FunctionType,
    },
}
