import types

test_types = {
    'function': {
        'args': tuple, 
        'kwargs': dict, 
    }, 
    '__main__': {
        'function': types.LambdaType,
        'dic': dict, 
        'tup': tuple, 
    }, 
}
