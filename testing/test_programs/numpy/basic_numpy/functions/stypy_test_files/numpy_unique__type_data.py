from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types.union_type import UnionType
from stypy.types.undefined_type import UndefinedType
import numpy

test_types = {
    '__main__': {
        'F': UnionType.create_from_type_list([numpy.int32, numpy.ndarray, numpy.ndarray]),
        '__builtins__': instance_of_class_name("module"),
        'h': instance_of_class_name("int"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'I': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        'r': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        'w': instance_of_class_name("int"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'n': instance_of_class_name("int"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
