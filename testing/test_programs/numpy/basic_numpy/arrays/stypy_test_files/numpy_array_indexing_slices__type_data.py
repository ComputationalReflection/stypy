from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'r4': instance_of_class_name("tuple"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'row_r2': instance_of_class_name("ndarray"),
        'row_r1': UnionType.create_from_type_list([numpy.int32, numpy.ndarray]),
        'r7': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),
        '__name__': instance_of_class_name("str"),
        'a': instance_of_class_name("ndarray"),
        'r5': instance_of_class_name("ndarray"),
        'r6': instance_of_class_name("tuple"),
        'col_r2': instance_of_class_name("ndarray"),
        'r1': UnionType.create_from_type_list([numpy.int32, numpy.ndarray]),
        'r2': UnionType.create_from_type_list([tuple, tuple]),
        'r3': instance_of_class_name("ndarray"),
        'r8': instance_of_class_name("tuple"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'col_r1': instance_of_class_name("ndarray"),

    },
}
