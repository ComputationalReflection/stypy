import types

test_types = {
    'f': {
        'y': int, 
        'x': int, 
        'z': int,
    }, 
    'f2': {
        'y': int, 
        'x': int, 
        'z': int,
    }, 
    'f3': {
        'y': bool, 
        'x': int,
        'z': str,
    }, 
    '__main__': {
        'f2': types.LambdaType, 
        'f3': types.LambdaType, 
        'f': types.LambdaType,
    }, 
}
