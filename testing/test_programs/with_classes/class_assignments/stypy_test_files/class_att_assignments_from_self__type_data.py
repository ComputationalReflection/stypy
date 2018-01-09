import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    'get_shape': {
        'self': instance_of_class_name("_ctypes"),
    },
    'get_strides': {
        'self': instance_of_class_name("_ctypes"),
    },
    'get_data': {
        'self': instance_of_class_name("_ctypes"),
    },
    '__init__': {
        'self': instance_of_class_name("_ctypes"),
    }, 
    '__main__': {
        'r4' : types.MethodType,
        'r2' : int,
        'r3' : int,
        'glob' : int,
        '_ctypes' : types.ClassType,
        'r' : int,
        'ra' : int,
        'rb' : int,
        'ct' : instance_of_class_name("_ctypes"),
    }, 
}
