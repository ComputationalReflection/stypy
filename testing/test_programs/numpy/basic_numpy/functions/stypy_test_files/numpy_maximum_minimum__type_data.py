from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'Zs': instance_of_class_name("ndarray"),
        'Rs': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'stop': instance_of_class_name("float"),
        'P': instance_of_class_name("ndarray"),
        'shape': instance_of_class_name("tuple"),
        'R': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        'fill': instance_of_class_name("int"),
        'R_start': instance_of_class_name("list"),
        '__package__': instance_of_class_name("NoneType"),
        'start': instance_of_class_name("float"),
        'r': instance_of_class_name("list"),
        'R_stop': instance_of_class_name("list"),
        'np': instance_of_class_name("module"),
        'position': instance_of_class_name("tuple"),
        'z': instance_of_class_name("list"),
        'Z_start': instance_of_class_name("list"),
        '__doc__': instance_of_class_name("NoneType"),
        'Z_stop': instance_of_class_name("list"),

    },
}
