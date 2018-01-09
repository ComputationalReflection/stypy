import types

from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'a': union_type.UnionType.create_from_type_list([int, str, Exception, UndefinedType]),
        'TypeDataFileWriter': types.ClassType,
        'z': types.NoneType,
    },
}
