from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'r4': instance_of_class_name("bool"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__name__': instance_of_class_name("str"),
        'rf3': UnionType.create_from_type_list([numpy.float32, numpy.float64, UndefinedType]),
        'rf2': UnionType.create_from_type_list([int, long, UndefinedType]),
        'rf1': UnionType.create_from_type_list([int, long, UndefinedType]),
        'rf5': UnionType.create_from_type_list([numpy.float32, numpy.float64, UndefinedType]),
        'r2': instance_of_class_name("float64"),
        'rf4': UnionType.create_from_type_list([numpy.float32, numpy.float64, UndefinedType]),
        'r3': instance_of_class_name("bool"),
        '__package__': instance_of_class_name("NoneType"),
        'r': UnionType.create_from_type_list([numpy.float32, numpy.float64]),
        'np': instance_of_class_name("module"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
