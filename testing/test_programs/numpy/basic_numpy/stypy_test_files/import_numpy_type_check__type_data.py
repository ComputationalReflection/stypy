import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'r': instance_of_class_name("ndarray"),
        'numpy.lib.type_check': types.ModuleType,
        'numpy.lib': types.ModuleType,
        'numpy': types.ModuleType,
    },
}
