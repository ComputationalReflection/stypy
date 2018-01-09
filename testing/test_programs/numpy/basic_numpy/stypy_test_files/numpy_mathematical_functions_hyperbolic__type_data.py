from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r12': instance_of_class_name("ndarray"),
        'r11': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__name__': instance_of_class_name("str"),
        'r4': instance_of_class_name("float64"),
        'r5': instance_of_class_name("float64"),
        'r6': instance_of_class_name("float64"),
        'r7': instance_of_class_name("ndarray"),
        'r1': instance_of_class_name("float64"),
        'r2': instance_of_class_name("float64"),
        'r3': instance_of_class_name("float64"),
        'r8': instance_of_class_name("ndarray"),
        'r9': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'x': instance_of_class_name("list"),
        'r10': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
