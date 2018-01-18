import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'BH': types.ClassType,
        'Body': types.ClassType,
        'Cell': types.ClassType,
        'HG': types.ClassType,
        'Node': types.ClassType,
        'Random': types.ClassType,
        'Tree': types.ClassType,
        'Vec3': types.ClassType,
        'argv': list,
        'clock': types.BuiltinFunctionType,
        'copy': types.FunctionType,
        'floor': types.BuiltinFunctionType,
        'maxint': int,
        'pi': float,
        'run': types.FunctionType,
        'sqrt': types.BuiltinFunctionType,
        'stderr': instance_of_class_name('FlushingStringIO'),
    },
}
