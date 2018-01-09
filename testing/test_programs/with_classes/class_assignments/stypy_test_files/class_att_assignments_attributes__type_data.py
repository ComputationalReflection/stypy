import types

test_types = {
    '__init__': {
        'self': types.InstanceType, 
    }, 
    '__main__': {
        'C': types.ClassType, 
        'c': types.InstanceType,
        'x': str,
    }, 
}
