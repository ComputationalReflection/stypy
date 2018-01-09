from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy
from stypy.types.union_type import UnionType
from stypy.invokation.type_rules.type_groups.type_groups import UndefinedType

test_types = {
    '__main__': {
        'a': instance_of_class_name("int"),
        'phi': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'y_int': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'r': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        '__name__': instance_of_class_name("str"),
        'x': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        'y': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        'r_int': instance_of_class_name("ndarray"),
        'x_int': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        'dr': UnionType.create_from_type_list([numpy.ndarray, UndefinedType]),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
