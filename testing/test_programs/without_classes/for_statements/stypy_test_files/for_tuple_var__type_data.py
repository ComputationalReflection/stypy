from stypy.types import union_type

test_types = {
    '__main__': {
        'arguments': dict, 
        'key': union_type.UnionType.create_from_type_list([int, str]),
        'arg': union_type.UnionType.create_from_type_list([int, str]),
        'ret_str': str, 
    }, 
}
