from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'np': instance_of_class_name("module"),
        'r': instance_of_class_name("tuple"),
        '__builtins__': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__file__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),
        '__package__': instance_of_class_name("NoneType"),

    },
}
