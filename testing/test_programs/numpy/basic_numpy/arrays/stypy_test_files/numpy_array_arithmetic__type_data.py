from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r4': instance_of_class_name("ndarray"),
        'r10': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'r6': instance_of_class_name("ndarray"),
        'r7': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'a': instance_of_class_name("ndarray"),
        'r5': instance_of_class_name("ndarray"),
        'c': instance_of_class_name("ndarray"),
        'b': instance_of_class_name("ndarray"),
        'r2': instance_of_class_name("ndarray"),
        'r3': instance_of_class_name("ndarray"),
        'j': instance_of_class_name("ndarray"),
        'r8': instance_of_class_name("ndarray"),
        'r9': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'r': instance_of_class_name("ndarray"),
        'np': instance_of_class_name("module"),
        '__doc__': instance_of_class_name("NoneType"),

    },
}
