import types

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.type_rules.type_groups.type_group_generator import DynamicType

test_types = {
    '__main__': {
        'TypeDataFileWriter': types.ClassType,
        'x': DynamicType,
        'y': StypyTypeError,
    },
}
