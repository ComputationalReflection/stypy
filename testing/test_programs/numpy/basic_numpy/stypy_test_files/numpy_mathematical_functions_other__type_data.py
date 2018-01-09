
import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
		'r4': instance_of_class_name("ndarray"),
		'r1': instance_of_class_name("ndarray"),
		'r2': instance_of_class_name("float64"),
		'r3': instance_of_class_name("ndarray"),
		'__builtins__': instance_of_class_name("module"),
		'__file__': instance_of_class_name("str"),
		'__package__': instance_of_class_name("NoneType"),
		'__name__': instance_of_class_name("str"),
		'np': instance_of_class_name("module"),
		'x': instance_of_class_name("list"),
		'__doc__': instance_of_class_name("NoneType"),
		

    },
}
