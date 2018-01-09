from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'distance': instance_of_class_name("ndarray"),
        'r1': instance_of_class_name("tuple"),
        'r2': instance_of_class_name("tuple"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'y': instance_of_class_name("ndarray"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
