from stypy.errors.type_error import StypyTypeError
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name

test_types = {
    '__main__': {
        'r4': instance_of_class_name("float64"),
        'r5': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'np': instance_of_class_name("module"),
        'x2': instance_of_class_name("ndarray"),
        '__name__': instance_of_class_name("str"),
        'y2': instance_of_class_name("ndarray"),
        'p2': instance_of_class_name("Polynomial"),
        'p3': instance_of_class_name("Chebyshev"),
        'r2': instance_of_class_name("ndarray"),
        'r3': instance_of_class_name("int"),
        't2': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'p': instance_of_class_name("poly1d"),
        'r': instance_of_class_name("float64"),
        't': instance_of_class_name("ndarray"),
        'y': instance_of_class_name("ndarray"),
        'x': instance_of_class_name("ndarray"),
        '__doc__': instance_of_class_name("NoneType"),
        'r6': StypyTypeError,

    },
}
