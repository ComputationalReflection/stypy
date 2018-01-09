import types

test_types = {
    'another_function': {
        'z': int, 
    }, 
    'function': {
        'x': int, 
        'another_function': types.LambdaType, 
    }, 
    '__main__': {
        'function': types.LambdaType, 
        'TypeDataFileWriter': types.ClassType, 
        'ret': str, 
    }, 
}
