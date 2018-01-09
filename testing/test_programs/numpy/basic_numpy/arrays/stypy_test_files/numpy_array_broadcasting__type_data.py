from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'i': instance_of_class_name("int"),
        '__file__': instance_of_class_name("str"),
        'np': instance_of_class_name("module"),
        '__package__': instance_of_class_name("NoneType"),
        '__builtins__': instance_of_class_name("module"),
        'r': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'v': instance_of_class_name("ndarray"),
        'y': instance_of_class_name("ndarray"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
