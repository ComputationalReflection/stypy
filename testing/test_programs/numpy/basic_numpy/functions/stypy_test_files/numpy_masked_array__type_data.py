from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r4': instance_of_class_name("MaskedArray"),
        'r2': instance_of_class_name("ndarray"),
        'r3': instance_of_class_name("MaskedArray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'r': instance_of_class_name("float64"),
        '__doc__': instance_of_class_name("NoneType"),
        '__name__': instance_of_class_name("str"),
        'x2': instance_of_class_name("ndarray"),
        'x': instance_of_class_name("ndarray"),
        'mx': instance_of_class_name("MaskedArray"),

    },
}
