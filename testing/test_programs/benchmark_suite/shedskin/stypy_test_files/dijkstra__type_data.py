import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'Edge':  types.ClassType,
        'Extract_Min': types.FunctionType,
        'Graph': types.ClassType,
        'Vertex': types.ClassType,
        'dijkstra': types.FunctionType,
        'random': types.ModuleType,
        'run': types.FunctionType,
    },
}
