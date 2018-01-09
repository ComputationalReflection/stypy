import types

from stypy.errors.type_error import StypyTypeError
from stypy.types import union_type
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        'theStr': str,
        'theInt': int,
        'union': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant1': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant2': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant3': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant4': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant5': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant6': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_variant7': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },
}
