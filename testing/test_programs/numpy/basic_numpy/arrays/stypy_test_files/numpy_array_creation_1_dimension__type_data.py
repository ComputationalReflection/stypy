from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r4': instance_of_class_name("int"),
        'r2': instance_of_class_name("dtype"),
        'F': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'r3': instance_of_class_name("int"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("dtype"),
        'V': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
