from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'a': instance_of_class_name("ndarray"),
        'b': instance_of_class_name("ndarray"),
        'r2': UnionType.create_from_type_list([numpy.int32, numpy.ndarray]),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'r': UnionType.create_from_type_list([numpy.int32, numpy.ndarray]),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
