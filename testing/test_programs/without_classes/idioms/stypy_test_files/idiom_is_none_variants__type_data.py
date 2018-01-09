import types

from stypy.errors.type_error import StypyTypeError
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'r4': int,
        'r5': float,
        'r2': int,
        'r3': float,
        'rd': types.NoneType,
        'rb': types.NoneType,
        'r': str,
        'rc': types.NoneType,
    },
}
