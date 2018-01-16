import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'EXIT': int,
        'FILE_':  int,
        'FilebasedMazeGame':  types.ClassType,
        'Maze':  types.ClassType,
        'MazeError':  types.ClassType,
        'MazeFactory':  types.ClassType,
        'MazeGame':  types.ClassType,
        'MazeReader':  types.ClassType,
        'MazeReaderException':  types.ClassType,
        'MazeSolver':  types.ClassType,
        'PATH':  int,
        'Relative':   types.FunctionType,
        'SOCKET':  int,
        'START':  int,
        'STDIN':  int,
        'os':   types.ModuleType,
        'random':   types.ModuleType,
        'run':   types.FunctionType,
        'sys':   types.ModuleType,
    },
}
