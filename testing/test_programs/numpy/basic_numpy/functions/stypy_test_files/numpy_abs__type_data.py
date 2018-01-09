from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'index': instance_of_class_name("int32"),
        'e': instance_of_class_name("int32"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'v': instance_of_class_name("float"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
