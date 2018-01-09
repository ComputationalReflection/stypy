from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'index': instance_of_class_name("tuple"),
        'r2': UnionType.create_from_type_list([tuple, UndefinedType]),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'value': instance_of_class_name("long"),
        '__package__': instance_of_class_name("NoneType"),
        'r': UnionType.create_from_type_list([tuple, UndefinedType]),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
