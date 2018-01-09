import types
from stypy.types import union_type

test_types = {
    '__main__': {
        'd': dict,
        'val': union_type.UnionType.create_from_type_list([str, int]),
        'cast': dict,
        'key': union_type.UnionType.create_from_type_list([str, int]),
    }, 
}
