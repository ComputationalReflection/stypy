from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'a': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'x': instance_of_class_name("list"),
        'x2': instance_of_class_name("ndarray"),
        'x3': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),
        'x4': instance_of_class_name("ndarray"),

    },
}
