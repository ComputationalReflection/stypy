import types

test_types = {
    'static': {
        'y': int, 
        'x': int, 
    }, 
    'instance': {
        'y': str, 
        'x': str, 
        'self': types.InstanceType, 
    }, 
    '__main__': {
        'r1': int, 
        'r2': str, 
        'f': types.InstanceType,
        'Foo': types.ClassType, 
    }, 
}
