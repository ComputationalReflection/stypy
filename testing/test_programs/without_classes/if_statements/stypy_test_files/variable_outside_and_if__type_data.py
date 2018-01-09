from stypy.types import union_type

test_types = {
    '__main__': {
        'c': bool, 
        'out_and_if': union_type.UnionType.create_from_type_list([str, int]),
        'result': union_type.UnionType.create_from_type_list([str, int]),
    }, 
}
