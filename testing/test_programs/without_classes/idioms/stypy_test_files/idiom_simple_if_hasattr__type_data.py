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

    'simple_if_hasattr_base1': {
        'r': int,
        'r2': StypyTypeError,
        'b': int,
        'r3': int,
        'r4': int,
        'a': int,
    },

    'simple_if_hasattr_base2': {
        'b': str,
        'r3': StypyTypeError,
        'r4': StypyTypeError,
        'a': str,
    },

    'simple_if_hasattr_base3': {
        'r': union_type.UnionType.create_from_type_list([int, UndefinedType]),
        'r2': UndefinedType,
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_hasattr_call_int': {
        'r': int,
        'r2': StypyTypeError,
        'b': int,
        'r3': int,
        'r4': int,
        'a': int,
    },

    'simple_if_hasattr_call_str': {
        'r3': int,
        'r': int,
        'r2': str,
        'b': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    # 'simple_if_idiom_variant': {
    #     'r': union_type.UnionType.create_from_type_list([UndefinedType, int]),
    #     'r2': UndefinedType,
    #     'b': union_type.UnionType.create_from_type_list([str, int]),
    #     'r3': int,
    #     'r4': int,
    #     'a': union_type.UnionType.create_from_type_list([str, int]),
    # },
    #
    'simple_if_not_hasattr_idiom': {
        'r': UndefinedType,
        'r2': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_hasattr_idiom_attr': {
        'r': int,
        'r2': StypyTypeError,
        'b': int,
        'r3': int,
        'r4': int,
        'a': types.InstanceType
    },

    'simple_if_hasattr_idiom_attr_b': {
        'b': int,
        'r3': StypyTypeError,
        'r4': int,
        'a': types.InstanceType,
    },
}
