from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'M': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'p': instance_of_class_name("int"),
        'S': instance_of_class_name("ndarray"),
        'V': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'n': instance_of_class_name("int"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
