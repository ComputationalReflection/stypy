import types

from stypy.errors import type_error
from stypy.types import union_type

test_types = {
    'functionb': {
        'a': int,
        'x': list,
    },
    '__main__': {
        'functionb': types.FunctionType,
        'r1': str,
        'r2': list,
    },
}
