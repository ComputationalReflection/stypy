import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'Bdecode': types.FunctionType,
        'Bencode': types.FunctionType,
        'N': int,
        'Relative': types.FunctionType,
        'compress_it': types.FunctionType,
        'dec_to_bin': types.FunctionType,
        'decode': types.FunctionType,
        'easytest': types.FunctionType,
        'encode': types.FunctionType,
        'f': float,
        'find_idx': types.FunctionType,
        'find_name': types.FunctionType,
        'findprobs': types.FunctionType,
        'hardertest': types.FunctionType,
        'internalnode': types.ClassType,
        'iterate': types.FunctionType,
        'makenodes': types.FunctionType,
        'node': types.ClassType,
        'os': types.ModuleType,
        'run': types.FunctionType,
        'sys': types.ModuleType,
        'test': types.FunctionType,
        'uncompress_it': types.FunctionType,
        'verbose': int,
        'weight': types.FunctionType,
    },
}
