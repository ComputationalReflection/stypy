import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'Chaosgame': types.ClassType,
        'GVector': types.ClassType,
        'GetKnots': types.FunctionType,
        'Relative': types.FunctionType,
        'Spline': types.ClassType,
        'main': types.FunctionType,
        'math': types.ModuleType,
        'os': types.ModuleType,
        'random': types.ModuleType,
        'run': types.FunctionType,
        'save_im': types.FunctionType,
        'sys': types.ModuleType,
        'time': types.ModuleType,
    },
}
