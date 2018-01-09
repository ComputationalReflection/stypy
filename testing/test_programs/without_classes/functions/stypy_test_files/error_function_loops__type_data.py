import types

from stypy.types import union_type
from stypy.errors import type_error

test_types = {
    'functionb': {
        'i': int,
        'x': list,
    },
    '__main__': {
        'functionb': types.FunctionType,

        'r1': str,
        'r2': list,
    },
}
