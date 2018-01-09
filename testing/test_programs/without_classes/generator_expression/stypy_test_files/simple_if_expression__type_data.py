import types
from stypy.types import union_type

test_types = {
    '__main__': {
        'a': int, 
        'b': int, 
        'TypeDataFileWriter': types.ClassType, 
        'x': int, 
        'y': int, 
        'z': union_type.UnionType.create_from_type_list([int, str]),
    }, 
}
