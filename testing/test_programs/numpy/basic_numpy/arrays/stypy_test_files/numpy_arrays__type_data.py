from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'a': instance_of_class_name("ndarray"),
        'r4': instance_of_class_name("ndarray"),
        'r6': instance_of_class_name("tuple"),
        'b': instance_of_class_name("ndarray"),
        'r2': instance_of_class_name("tuple"),
        'r3': instance_of_class_name("tuple"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("type"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'r5': instance_of_class_name("tuple"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
