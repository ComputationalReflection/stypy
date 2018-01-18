import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'bestPath': types.FunctionType,
        'doSumWeight': types.FunctionType,
        'evaporatePher': types.FunctionType,
        'findSumWeight': types.FunctionType,
        'genPath': types.FunctionType,
        'main': types.FunctionType,
        'pathLength': types.FunctionType,
        'random': types.ModuleType,
        'randomMatrix': types.FunctionType,
        'run': types.FunctionType,
        'updatePher': types.FunctionType,
        'wrappedPath': types.FunctionType,
    },
}
