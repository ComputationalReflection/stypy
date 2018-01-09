import types
from stypy.errors.type_error import StypyTypeError

test_types = {
    'met': {
        'self': types.InstanceType, 
    }, 
    '__main__': {
        'func_predelete': types.MethodType, 
        'met_predelete': types.MethodType, 
        'f': types.InstanceType, 
        'att_predelete': str, 
        'met_result_predelete': str, 
        'TypeDataFileWriter': types.ClassType, 
        'Foo': types.ClassType,
        'att_postdelete': StypyTypeError,
        'met_result_postdelete': StypyTypeError,
        'met_postdelete': StypyTypeError,
        'func_result_postdelete': StypyTypeError,
    }, 
}
