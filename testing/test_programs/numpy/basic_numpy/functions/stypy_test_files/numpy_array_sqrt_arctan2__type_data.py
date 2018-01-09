from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'R': instance_of_class_name("ndarray"),
        'T': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'Y': instance_of_class_name("ndarray"),
        'X': instance_of_class_name("ndarray"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
