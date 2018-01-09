from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    'pure_python_version': {
        'i': instance_of_class_name("int"),
        'Y': instance_of_class_name("list"),
        'Z': instance_of_class_name("list"),
        'X': instance_of_class_name("list"),
        't1': instance_of_class_name("float"),
    },
    'numpy_version': {
        'Y': instance_of_class_name("ndarray"),
        'X': instance_of_class_name("ndarray"),
        'Z': instance_of_class_name("ndarray"),
        't1': instance_of_class_name("float"),
    },
    '__main__': {
        '__builtins__': instance_of_class_name("module"),
        'size_of_vec': instance_of_class_name("int"),
        '__file__': instance_of_class_name("str"),
        't2': instance_of_class_name("float"),
        'pure_python_version': instance_of_class_name("function"),
        '__package__': instance_of_class_name("NoneType"),
        'numpy_version': instance_of_class_name("function"),
        'time': instance_of_class_name("module"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        't1': instance_of_class_name("float"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
