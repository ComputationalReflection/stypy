from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'NamedArray': instance_of_class_name("type"),
        'r': instance_of_class_name("str"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("NamedArray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
