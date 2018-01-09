from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'r4': instance_of_class_name("ndarray"),
        'r5': instance_of_class_name("ndarray"),
        'r6': instance_of_class_name("ndarray"),
        'r2': instance_of_class_name("int32"),
        'r3': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("int32"),
        'w': instance_of_class_name("ndarray"),
        'v': instance_of_class_name("ndarray"),
        'y': instance_of_class_name("ndarray"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
