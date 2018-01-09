from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r4': instance_of_class_name("ndarray"),
        'r5': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'r2': instance_of_class_name("ndarray"),
        'r3': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("ndarray"),
        'w': instance_of_class_name("ndarray"),
        'v': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
