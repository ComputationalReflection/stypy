import types

test_types = {
    '__init__': {
        'self': types.InstanceType, 
    }, 
    'method': {
        'self': types.InstanceType, 
    }, 
    '__main__': {
        'C': types.ClassType, 
        'c': types.InstanceType, 
        'TypeDataFileWriter': types.ClassType, 
        'x': bool,
        'obj': types.InstanceType,
        'sum': float,
    }, 
}
