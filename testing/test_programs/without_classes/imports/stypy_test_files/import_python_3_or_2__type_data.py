import types
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
test_types = {
    '__main__': {
        'builtins': types.ModuleType,
        'basestring': union_type.UnionType.create_from_type_list([type, UndefinedType]),
        'sys': types.ModuleType,
        'pickle': union_type.UnionType.create_from_type_list([types.ModuleType, types.ModuleType]),
    }, 
}
