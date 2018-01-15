import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'BETA0': int,
        'BETA1': int,
        'HALF': int,
        'M': int,
        'ONE': int,
        'QUARTER': int,
        'Relative': types.FunctionType,
        'THREEQU': int,
        'clear': types.FunctionType,
        'decode': types.FunctionType,
        'encode': types.FunctionType,
        'hardertest': types.FunctionType,
        'os': types.ModuleType,
        'run': types.FunctionType,
        'test': types.FunctionType,
    },
}
