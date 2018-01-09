import types
from stypy.errors.type_error import StypyTypeError

test_types = {
    'static': {
        'y': str,
        'x': str,
    },
    '__main__': {
        'r1': StypyTypeError,
        'r2': str, 
        'f': types.InstanceType,
        'Foo': types.ClassType, 
    }, 
}
