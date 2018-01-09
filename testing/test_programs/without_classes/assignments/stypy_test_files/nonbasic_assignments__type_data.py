import types
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'a': types.BuiltinFunctionType,
        'b': int,
        'c': UndefinedType,
        'l': list, 
        'TypeDataFileWriter': types.ClassType, 
    }, 
}
