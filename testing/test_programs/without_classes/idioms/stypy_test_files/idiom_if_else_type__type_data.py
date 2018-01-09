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

    'if_else_base1': {
        'r': int,
        'r2': StypyTypeError,
        'b': int,
        'a': int,
    },

    'if_else_base2': {
        'r3': str,
        'r4': StypyTypeError,
        'b': str,
        'a': str,
        'r5': StypyTypeError,
        'r6': StypyTypeError,
    },

    'if_else_base3': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'r3': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'r4': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'a': union_type.UnionType.create_from_type_list([str, int]),
        'r5': int,
        'r6': int,
    },

    'if_else_base4': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'r3': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'r4': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'a': union_type.UnionType.create_from_type_list([str, bool, int]),
        'r5': int,
        'r6': int,
    },

    'simple_if_else_idiom_variant': {
        'r': union_type.UnionType.create_from_type_list([UndefinedType, int]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'r4': UndefinedType,
        'a': union_type.UnionType.create_from_type_list([str, int]),
        'r5': int,
        'r6': int,
    },

    'simple_if_else_not_idiom': {
        'r': union_type.UnionType.create_from_type_list([UndefinedType, int]),
        'r2': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'r4': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r5': int,
        'r6': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_else_idiom_attr': {
        'r': int,
        'r2': StypyTypeError,
        'b': int,
        'a': types.InstanceType
    },

    'simple_if_else_idiom_attr_b': {
        'b': str,
        'r3': StypyTypeError,
        'r4': StypyTypeError,
        'a': types.InstanceType,
    },
}
