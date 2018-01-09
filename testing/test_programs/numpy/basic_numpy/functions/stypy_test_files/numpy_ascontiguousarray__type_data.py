from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types.union_type import UnionType
import numpy

test_types = {
    '__main__': {
        'idx': UnionType.create_from_type_list([numpy.ndarray, numpy.ndarray]),
        'uZ': numpy.ndarray,
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'T': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),
        '_': UnionType.create_from_type_list([numpy.ndarray, numpy.ndarray]),

    },
}
