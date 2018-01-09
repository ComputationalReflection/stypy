import types
from stypy.types import union_type

test_types = {
    'inc': {
        'self': types.InstanceType, 
        'value': float,
    }, 
    'bitwise_or': {
        'x': union_type.UnionType.create_from_type_list([float, int]),
        'n': int, 
    }, 
    'flow_sensitive': {
        'condition': bool, 
    }, 
    '__main__': {
        'Counter': types.ClassType, 
        'obj': types.InstanceType, 
        'flow_sensitive': types.LambdaType, 
        'bitwise_or': types.LambdaType,
    }, 
}
