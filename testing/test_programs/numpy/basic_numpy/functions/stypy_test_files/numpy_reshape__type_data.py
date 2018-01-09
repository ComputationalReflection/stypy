from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'A': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        '__doc__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'sum': instance_of_class_name("ndarray"),

    },
}
