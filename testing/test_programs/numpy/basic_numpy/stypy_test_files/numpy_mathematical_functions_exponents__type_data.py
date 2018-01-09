
import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
		'r16': instance_of_class_name("ndarray"),
		'r17': instance_of_class_name("ndarray"),
		'r14': instance_of_class_name("ndarray"),
		'r15': instance_of_class_name("ndarray"),
		'r12': instance_of_class_name("ndarray"),
		'r13': instance_of_class_name("ndarray"),
		'r10': instance_of_class_name("ndarray"),
		'r11': instance_of_class_name("ndarray"),
		'__builtins__': instance_of_class_name("module"),
		'__file__': instance_of_class_name("str"),
		'r18': instance_of_class_name("ndarray"),
		'x2': instance_of_class_name("list"),
		'__name__': instance_of_class_name("str"),
		'x1': instance_of_class_name("list"),
		'r4': instance_of_class_name("float64"),
		'r5': instance_of_class_name("float64"),
		'r6': instance_of_class_name("float64"),
		'r7': instance_of_class_name("float64"),
		'r1': instance_of_class_name("float64"),
		'r2': instance_of_class_name("float64"),
		'r3': instance_of_class_name("float64"),
		'r8': instance_of_class_name("float64"),
		'r9': instance_of_class_name("float64"),
		'__package__': instance_of_class_name("NoneType"),
		'np': instance_of_class_name("module"),
		'x': instance_of_class_name("list"),
		'__doc__': instance_of_class_name("NoneType"),
		

    },
}
