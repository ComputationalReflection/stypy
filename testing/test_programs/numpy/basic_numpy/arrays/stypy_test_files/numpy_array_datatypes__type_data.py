from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'r2': instance_of_class_name("dtype"),
        'r3': instance_of_class_name("dtype"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("dtype"),
        '__name__': instance_of_class_name("str"),
        'np': instance_of_class_name("module"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
