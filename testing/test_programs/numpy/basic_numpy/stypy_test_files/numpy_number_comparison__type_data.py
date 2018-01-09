from stypy.errors.type_error import StypyTypeError
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types import union_type, undefined_type

test_types = {
    '__main__': {
        'r14': instance_of_class_name("ndarray"),
        'r15': instance_of_class_name("ndarray"),
        'r12': instance_of_class_name("ndarray"),
        'r13': instance_of_class_name("ndarray"),
        'r10': instance_of_class_name("ndarray"),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        '__name__': instance_of_class_name("str"),
        'Z': instance_of_class_name("ndarray"),
        'r1': union_type.UnionType.create_from_type_list([float, undefined_type.UndefinedType]),
        'r4': union_type.UnionType.create_from_type_list([float, undefined_type.UndefinedType]),
        'r5': instance_of_class_name("bool"),
        'r6': instance_of_class_name("ndarray"),
        'r7': instance_of_class_name("ndarray"),
        'r2': union_type.UnionType.create_from_type_list([bool, undefined_type.UndefinedType]),
        'r3': union_type.UnionType.create_from_type_list([bool, undefined_type.UndefinedType]),
        'r8': instance_of_class_name("ndarray"),
        'r9': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        '__doc__': instance_of_class_name("NoneType"),
        #'r11': StypyTypeError,

    },
}
