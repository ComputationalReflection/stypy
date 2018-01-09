import types
from stypy.errors.type_error import StypyTypeError

test_types = {
    '__main__': {
        'y2' : StypyTypeError,
        'x1' : int,
        'f' : types.InstanceType,
        'Nested' : types.ClassType,
        'y1' : types.InstanceType,
        'x2' : StypyTypeError,
        'Foo' : types.ClassType,
    },
}
