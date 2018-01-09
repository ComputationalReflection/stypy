import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': types.InstanceType,
    },
    '__main__': {
        'r1': tuple,
        'r2': tuple,
        'r3': tuple,
        'f': types.InstanceType,
        'Foo': types.ClassType,
        '_mapper': list,
    }, 
}
