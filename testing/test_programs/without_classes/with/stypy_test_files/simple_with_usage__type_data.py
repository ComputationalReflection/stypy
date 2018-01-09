import types

test_types = {
    '__enter__': {
        'self': types.InstanceType, 
    }, 
    '__exit__': {
        'self': types.InstanceType, 
        'traceback': types.NoneType, 
        'type': types.NoneType, 
        'value': types.NoneType, 
    }, 
    '__main__': {
        'a': int, 
        'thing': int, 
        'TypeDataFileWriter': types.ClassType, 
        'controlled_execution': types.ClassType, 
    }, 
}
