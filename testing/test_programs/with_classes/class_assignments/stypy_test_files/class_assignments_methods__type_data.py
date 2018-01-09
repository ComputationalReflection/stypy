import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': instance_of_class_name("ndenumerate"),
    }, 
    '__main__': {
        'r2': types.MethodType,
        'r': types.MethodType,
        'ndenumerate': types.ClassType,
        'o': instance_of_class_name("ndenumerate"),
    }, 
}
