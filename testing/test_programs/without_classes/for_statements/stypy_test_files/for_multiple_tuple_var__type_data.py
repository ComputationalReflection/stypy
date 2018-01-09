from stypy.types import union_type

test_types = {
    '__main__': {
        'axis': union_type.UnionType.create_from_type_list([int, tuple]),
        'pad_before': union_type.UnionType.create_from_type_list([int, tuple]),
        'pad_after': union_type.UnionType.create_from_type_list([int, tuple]),
        'before_val': union_type.UnionType.create_from_type_list([int, tuple]),
        'after_val': union_type.UnionType.create_from_type_list([int, tuple]),
    }, 
}
