import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': instance_of_class_name("StringConverter"),
    },
    'asbytes': {
        'st': str,
    }, 
    '__main__': {
        'r4': tuple,
        'r': list,
        'r2': tuple,
        'r3': tuple,
        'StringConverter': types.ClassType,
        'st': instance_of_class_name("StringConverter"),
        'asbytes': types.FunctionType,
    }, 
}
