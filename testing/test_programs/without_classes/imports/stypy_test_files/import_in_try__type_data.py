import types
from stypy.types.union_type import UnionType
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'fabs': UnionType.create_from_type_list([types.BuiltinFunctionType, UndefinedType]),
        'cos': UnionType.create_from_type_list([types.BuiltinFunctionType, UndefinedType]),
        'ex': UnionType.create_from_type_list([Exception, UndefinedType]),
        'r': float,
        'kos': UndefinedType,
        'r2': UndefinedType,
        'sin': UndefinedType,
    }, 
}
