
"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'redon'
2: 

"""

# Import the stypy library
from stypy import *

# Create the module type store
type_store = TypeStore(__file__)

################## Begin of the type inference program ##################


# Assignment to a Name from a Str
__temp_1998 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 1, 13), 'str', 'redon')
# Type assignment (line 1)
type_store.set_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 1, 0), '__author__', __temp_1998)

################## End of the type inference program ##################

module_errors = stypy.errors.type_error.TypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
