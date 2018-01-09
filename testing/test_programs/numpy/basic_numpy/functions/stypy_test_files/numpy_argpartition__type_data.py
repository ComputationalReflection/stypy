from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'Z': instance_of_class_name("ndarray"),
        'r2': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'r': instance_of_class_name("ndarray"),
        'x2': instance_of_class_name("NoneType"),
        '__name__': instance_of_class_name("str"),
        'n': instance_of_class_name("int"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
