from stypy.errors.type_error import StypyTypeError
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r5': instance_of_class_name("ndarray"),
        'r6': instance_of_class_name("ndarray"),
        'r7': instance_of_class_name("tuple"),
        'r1': instance_of_class_name("bool_"),
        'r2': instance_of_class_name("float64"),
        'r3': instance_of_class_name("tuple"),
        'r4': StypyTypeError,
        'r8': StypyTypeError,
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__name__': instance_of_class_name("str"),
        'x2': instance_of_class_name("list"),
        'x': instance_of_class_name("list"),
        'x1': instance_of_class_name("list"),
        '__doc__': instance_of_class_name("NoneType"),
    },
}
