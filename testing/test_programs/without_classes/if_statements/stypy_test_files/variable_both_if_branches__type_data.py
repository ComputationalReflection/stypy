from stypy.types import union_type

test_types = {
    '__main__': {
        'a': int,
        'x': union_type.UnionType.create_from_type_list([str, int]),
        'c': str,
    }, 
}
