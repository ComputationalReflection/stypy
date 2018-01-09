import types
from stypy.types import union_type
from stypy.types import undefined_type
test_types = {
    'formatargspec': {
        'formatvalue': types.FunctionType,
        'join': types.FunctionType,
        'formatvarkw': types.FunctionType,
        'i': union_type.UnionType.create_from_type_list([int, undefined_type.UndefinedType]),
        'args': tuple,
        'formatarg': str,
        'firstdefault': union_type.UnionType.create_from_type_list([int, undefined_type.UndefinedType]),
        'defaults': tuple,
        'varargs': types.NoneType,
        'formatvarargs': types.FunctionType,
        'varkw': types.NoneType,
        'specs': list
    },
    '__main__': {
        'joinseq': types.FunctionType,
        'formatargspec': types.FunctionType,
        'strseq': types.FunctionType,
        'foo': types.FunctionType,
        'foo2': types.FunctionType,
        '__package__': None,
        '__doc__': None,
        '__file__': str,
        '__name__': str,
        'r': str,
    },
}