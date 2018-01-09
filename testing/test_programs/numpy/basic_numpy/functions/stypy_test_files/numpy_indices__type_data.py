from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'cartesian': instance_of_class_name("function"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
