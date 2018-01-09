
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Use an unexisting module member"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7:     # Type error
8:     print math.non_existing
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Use an unexisting module member')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    
    # Getting the type of 'math' (line 8)
    math_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'math')
    # Obtaining the member 'non_existing' of a type (line 8)
    non_existing_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), math_2, 'non_existing')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
