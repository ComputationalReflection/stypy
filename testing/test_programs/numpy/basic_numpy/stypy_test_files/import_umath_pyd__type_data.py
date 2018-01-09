import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
test_types = {
    '__main__': {
        'x': instance_of_class_name("ufunc"),
        'y': instance_of_class_name("float64"),
        'numpy.core': types.ModuleType,
        'numpy.core.umath': types.ModuleType,
        'numpy': types.ModuleType,
    },
}
