from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'row1': instance_of_class_name("ndarray"),
        'row2': instance_of_class_name("ndarray"),
        'b': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'width': instance_of_class_name("int"),
        'r': instance_of_class_name("tuple"),
        'u': instance_of_class_name("int"),
        'board': instance_of_class_name("ndarray"),
        'w': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
