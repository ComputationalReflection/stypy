import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'AMINOACIDS': str,
        'BLIND': int,
        'CYTOSOLIC': int,
        'D': float,
        'EXTRACELLULAR': int,
        'LENGTH': int,
        'MITOCHONDRIAL': int,
        'NUCLEAR': int,
        'PROTEINS': list,
        'Protein': types.ClassType,
        'Relative': types.FunctionType,
        'calculate_error': types.FunctionType,
        'create_kernel_table': types.FunctionType,
        'create_tables': types.FunctionType,
        'exp': types.BuiltinFunctionType,
        'load_file': types.FunctionType,
        'main': types.FunctionType,
        'os': types.ModuleType,
        'run': types.FunctionType,
        'sys': types.ModuleType,
        'train_adatron': types.FunctionType,
    },
}
