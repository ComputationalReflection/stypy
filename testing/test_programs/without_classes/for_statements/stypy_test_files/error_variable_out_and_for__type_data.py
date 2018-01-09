from stypy.types import union_type

test_types = {
    '__main__': {
        'acum': union_type.UnionType.create_from_type_list([str, int]),
        'i': int,
    }, 
}
