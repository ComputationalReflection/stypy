import types

from stypy.types import union_type

test_types = {
    '__main__': {
        'bigUnion': union_type.UnionType.create_from_type_list([str, bool, int]),
        'theComplex': complex,
        'theStr': str,
        'intOrBool': union_type.UnionType.create_from_type_list([bool, int]),
        'union': union_type.UnionType.create_from_type_list([str, int]),
        'theInt': int,
        'intStrComplex': union_type.UnionType.create_from_type_list([str, complex, int]),
        'idiom': types.LambdaType,
        'theBool': bool,
        'r4': bool,
        'r5': bool,
        'r6': union_type.UnionType.create_from_type_list([bool, int]),
        'r7': union_type.UnionType.create_from_type_list([str, bool, int]),
        'r2': str,
        'r3': union_type.UnionType.create_from_type_list([str, int]),
        'r8': union_type.UnionType.create_from_type_list([str, bool, int]),
        'r': int,
    },
}
