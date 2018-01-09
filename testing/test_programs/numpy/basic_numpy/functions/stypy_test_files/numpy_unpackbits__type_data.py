from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types.union_type import UnionType
import numpy

test_types = {
    '__main__': {
        'B': instance_of_class_name("ndarray"),
        'r2': numpy.ndarray,
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'I': numpy.ndarray,
        'r': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
