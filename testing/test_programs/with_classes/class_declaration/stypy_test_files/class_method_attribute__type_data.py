import types
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__init__': {
        'self': types.InstanceType,
    },
    '__main__': {
        'y': str,
        'm': types.InstanceType,
        'MaskedArray': types.ClassType,
    }, 
}
