from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'Zmax': instance_of_class_name("float64"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        '__doc__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        'Zmin': instance_of_class_name("float64"),
        'Z2': instance_of_class_name("ndarray"),

    },
}
