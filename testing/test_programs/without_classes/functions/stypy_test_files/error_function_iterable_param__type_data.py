import types

from stypy.errors import type_error

test_types = {
    '__main__': {
        'fun1': types.FunctionType,
        'fun2': types.FunctionType,
        'S': list,
        'V': list,
        'normal_list': list,
        'tuple_': tuple,

        'r1': type_error.StypyTypeError,
        'r2': type_error.StypyTypeError,
        'r3': type_error.StypyTypeError,
        'r4': type_error.StypyTypeError,
        'r5': type_error.StypyTypeError,
    },
}
