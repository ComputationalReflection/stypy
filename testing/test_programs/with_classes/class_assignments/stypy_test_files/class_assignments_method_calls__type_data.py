import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': instance_of_class_name("TestCase"),
    },
    '_deprecate': {
        'deprecated_func': types.FunctionType,
        'original_func': types.MethodType,
    }, 
    '__main__': {
        'r2': types.MethodType,
        'r': types.MethodType,
        'TestCase': types.ClassType,
        't': instance_of_class_name("TestCase"),
    }, 
}
