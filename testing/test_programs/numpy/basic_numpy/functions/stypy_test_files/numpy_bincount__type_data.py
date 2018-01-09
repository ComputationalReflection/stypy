from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'D_means': instance_of_class_name("ndarray"),
        'D': instance_of_class_name("ndarray"),
        'F': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'D_counts': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'D_sums': instance_of_class_name("ndarray"),
        'S': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'I': instance_of_class_name("list"),
        'X': instance_of_class_name("list"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
