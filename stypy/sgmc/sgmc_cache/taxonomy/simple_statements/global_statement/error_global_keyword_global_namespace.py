
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Usage of the global keyword in the global namespace"
4: 
5: if __name__ == '__main__':
6:     global gvar
7: 
8:     # Type error
9:     print gvar
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Usage of the global keyword in the global namespace')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Marking variables as global (line 6)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 6, 4), 'gvar')
    # Getting the type of 'gvar' (line 9)
    gvar_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'gvar')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
