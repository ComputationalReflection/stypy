from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
from stypy.types import union_type, undefined_type
from numpy import ndarray

test_types = {
    '__main__': {
        'r23': instance_of_class_name("ndarray"),
        'r22': instance_of_class_name("ndarray"),
        'r21': instance_of_class_name("ndarray"),
        'r20': instance_of_class_name("ndarray"),
        '__package__': instance_of_class_name("NoneType"),
        'np': instance_of_class_name("module"),
        'o2': instance_of_class_name("ndarray"),
        'o1': instance_of_class_name("ndarray"),
        'r16': instance_of_class_name("ndarray"),
        'r17': instance_of_class_name("ndarray"),
        'r14': instance_of_class_name("ndarray"),
        'r15': instance_of_class_name("ndarray"),
        'r12': instance_of_class_name("float64"),
        'r13': instance_of_class_name("float64"),
        'r10': instance_of_class_name("float64"),
        'r11': union_type.UnionType.create_from_type_list([ndarray, undefined_type.UndefinedType]),
        '__builtins__': instance_of_class_name("module"),
        '__file__': instance_of_class_name("str"),
        'r18': instance_of_class_name("ndarray"),
        'r19': instance_of_class_name("ndarray"),
        'phase': union_type.UnionType.create_from_type_list([ndarray, undefined_type.UndefinedType, tuple]),
        '__name__': instance_of_class_name("str"),
        'r4': instance_of_class_name("float64"),
        'r5': instance_of_class_name("float64"),
        'r6': instance_of_class_name("float64"),
        'r7': instance_of_class_name("ndarray"),
        'r1': instance_of_class_name("float64"),
        'r2': instance_of_class_name("float64"),
        'r3': instance_of_class_name("float64"),
        'r8': union_type.UnionType.create_from_type_list([ndarray, undefined_type.UndefinedType]),
        'r9': instance_of_class_name("float64"),
        '__doc__': instance_of_class_name("NoneType"),
        'x10': instance_of_class_name("list"),
        'x': instance_of_class_name("list"),

    },
}
