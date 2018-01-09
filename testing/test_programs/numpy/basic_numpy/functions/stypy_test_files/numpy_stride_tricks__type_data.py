from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    'rolling': {
        'a': instance_of_class_name("ndarray"),
        'strides': instance_of_class_name("tuple"),
        'shape': instance_of_class_name("tuple"),
        'window': instance_of_class_name("int"),
    },
    '__main__': {
        'stride_tricks': instance_of_class_name("module"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'rolling': instance_of_class_name("function"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
