import types

from stypy.types import union_type

test_types = {
    'function': {
        'a': int,
    },
    '__main__': {
        'function': types.LambdaType,
        'TypeDataFileWriter': types.ClassType,
        'x': union_type.UnionType.create_from_type_list([str, int, bool]),
    },
}
