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

    'simple_if_not_base1': {
        'b': str,
        'r3': int,
        'r4': StypyTypeError,
        'a': int,
    },

    'simple_if_not_base2': {
        'b': int,
        'r': StypyTypeError,
        'r2': str,
        'r3': StypyTypeError,
        'r4': int,
        'a': str,
    },

    'simple_if_not_base3': {
        'r': UndefinedType,
        'r2': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },
    #
    'simple_if_not_call_int': {
        'b': str,
        'r3': int,
        'r4': StypyTypeError,
        'a': int,
    },

    'simple_if_not_call_str': {
        'b': int,
        'r2': str,
        'r': int,
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_not_idiom_variant': {
        'r': UndefinedType,
        'r2': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_not_not_idiom': {
        'r': union_type.UnionType.create_from_type_list([UndefinedType, int]),
        'r2': union_type.UnionType.create_from_type_list([str, UndefinedType]),
        'b': union_type.UnionType.create_from_type_list([str, int]),
        'r3': int,
        'r4': int,
        'a': union_type.UnionType.create_from_type_list([str, int]),
    },

    'simple_if_not_idiom_attr': {
        'b': str,
        'r3': int,
        'r4': StypyTypeError,
        'a': types.InstanceType
    },

    'simple_if_not_diom_attr_b': {
        'r': int,
        'r2': str,
        'b': int,
        'r3': StypyTypeError,
        'r4': int,
        'a': types.InstanceType,
    },
}
