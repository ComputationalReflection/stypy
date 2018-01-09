import types
from stypy.types import union_type

test_types = {
    'func': {
        'instance1b': union_type.UnionType.create_from_type_list([str, int, float]),
        'Type1b': union_type.UnionType.create_from_type_list([str, int, float]),
        'Type2b': union_type.UnionType.create_from_type_list([str, int, float]),
        'test1b': union_type.UnionType.create_from_type_list([str, tuple, tuple]),
        'instance2b': union_type.UnionType.create_from_type_list([str, int, float]),
        'test2b': union_type.UnionType.create_from_type_list([str, tuple, tuple]),
        '_': union_type.UnionType.create_from_type_list([str, int, float]),
    },
    '__main__': {
        'test1': union_type.UnionType.create_from_type_list([str, tuple]),
        'test2': union_type.UnionType.create_from_type_list([str, tuple]),
        'Type2b': union_type.UnionType.create_from_type_list([str, int, float]),
        'getlist': types.FunctionType,
        'instance1b': union_type.UnionType.create_from_type_list([str, int, float]),
        'instance2b': union_type.UnionType.create_from_type_list([str, int, float]),
        'test1b': union_type.UnionType.create_from_type_list([str, tuple, tuple]),
        '__package__': None,
        '__doc__': None,
        '__file__': str,
        'c': int,
        'Type1b': union_type.UnionType.create_from_type_list([str, int, float]),
        'instance2': union_type.UnionType.create_from_type_list([str, float]),
        'instance1': union_type.UnionType.create_from_type_list([str, int]),
        'func': types.FunctionType,
        '__name__': str,
        '_': union_type.UnionType.create_from_type_list([str, int, float]),
        'a': tuple,
        'b': int,
        'test2b': union_type.UnionType.create_from_type_list([str, tuple, tuple]),
        'Type1': union_type.UnionType.create_from_type_list([str, int]),
        'Type2': union_type.UnionType.create_from_type_list([str, float]),
    }, 
}
