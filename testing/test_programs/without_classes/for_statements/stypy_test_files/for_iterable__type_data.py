from stypy.types import union_type

test_types = {
    '__main__': {
        'l': list,
        's': str, 
        'elem': union_type.UnionType.create_from_type_list([int, str, bool]),
    }, 
}
