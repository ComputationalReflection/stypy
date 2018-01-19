import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {
    '__main__': {
        'alphaBeta': types.FunctionType,
        'alphaBetaQui': types.FunctionType,
        'bishopLines': tuple,
        'clearCastlingOpportunities': list,
        'copy': types.FunctionType,
        'evaluate': types.FunctionType,
        'i': union_type.UnionType.create_from_type_list([int, tuple]),
        'iFalse': int,
        'iNone': int,
        'iTrue': int,
        'kingMoves': tuple,
        'knightMoves': tuple,
        'legalMoves': types.FunctionType,
        'linePieces': tuple,
        'move': types.FunctionType,
        'moveStr': types.FunctionType,
        'nonpawnAttacks': types.FunctionType,
        'nonpawnBlackAttacks': types.FunctionType,
        'nonpawnWhiteAttacks': types.FunctionType,
        'pieces': str,
        'printBoard': types.FunctionType,
        'pseudoLegalCaptures': types.FunctionType,
        'pseudoLegalCapturesBlack': types.FunctionType,
        'pseudoLegalCapturesWhite': types.FunctionType,
        'pseudoLegalMoves': types.FunctionType,
        'pseudoLegalMovesBlack': types.FunctionType,
        'pseudoLegalMovesWhite': types.FunctionType,
        'queenLines': tuple,
        'rookLines': tuple,
        'rowAttack': types.FunctionType,
        'run': types.FunctionType,
        'setup': tuple,
        'speedTest': types.FunctionType,
        'squares': tuple,
        'toString': types.FunctionType,
        'v':  union_type.UnionType.create_from_type_list([int, tuple]),
    },
}
