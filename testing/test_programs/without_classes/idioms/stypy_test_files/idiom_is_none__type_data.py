import types

from stypy.errors.type_error import StypyTypeError
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'r2': int,
        'r3': float,
        'sys': types.ModuleType,
        'r': str,
        'rb': types.NoneType,
        'test_package': types.FunctionType,
        'os': types.ModuleType,
        'types': types.ModuleType,
    },
}
