from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types import union_type, undefined_type

test_types = {
    '__main__': {
        'r1': union_type.UnionType.create_from_type_list([str, undefined_type.UndefinedType]),
        'r2': instance_of_class_name("NoneType"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),
    },
}
