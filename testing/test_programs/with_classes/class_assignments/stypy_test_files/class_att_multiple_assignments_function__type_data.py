import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': types.InstanceType,
    },
    'my_zip': {
        'self': types.InstanceType,
        'list_': list,
    }, 
    '__main__': {
        'r4': tuple,
        'mapper': list,
        'C': types.ClassType,
        'r2': tuple,
        'r3': tuple,
        'c': types.InstanceType,
        'r': tuple,
        'Test': types.ClassType,
        'r5': tuple,
        'r6': tuple,

    }, 
}
