import types
from stypy.types import union_type

u_class = union_type.UnionType(types.ClassType, types.ClassType)
u_inst = union_type.UnionType(types.InstanceType, types.InstanceType)

test_types = {

    '__main__': {
        'A' : types.ClassType,
        'B' : types.ClassType,
        'Simple' : u_class,
        'base' : u_class,
        'w' : str,
        'x' : u_inst,
        'y' : str,
        '__name__' : str,
        'z' : str,
        'k' : int,
    }, 
}
