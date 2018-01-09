from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.types.undefined_type import UndefinedType

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': numpy.ndarray,
        '__doc__': instance_of_class_name("NoneType"),
        'Z2': UnionType.create_from_type_list([numpy.ndarray, UndefinedType, numpy.ndarray]),

    },
}
