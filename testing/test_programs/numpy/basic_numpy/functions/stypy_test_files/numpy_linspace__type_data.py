from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'spacing': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        '__name__': instance_of_class_name("str"),
        'r4': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        'r5': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        'r6': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        'r2': instance_of_class_name("ndarray"),
        'r3': instance_of_class_name("ndarray"),
        'samples3': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        'samples2': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("ndarray"),
        'samples': UnionType.create_from_type_list([numpy.ndarray, numpy.float64]),
        'np': instance_of_class_name("module"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
