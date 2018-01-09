
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types import union_type, undefined_type
from numpy import ndarray
import types

test_types = {
    '__main__': {
        'f': types.LambdaType,
        'sol': instance_of_class_name("scipy.optimize.optimize.OptimizeResult"),
        'x': instance_of_class_name("ndarray"),
    }
}