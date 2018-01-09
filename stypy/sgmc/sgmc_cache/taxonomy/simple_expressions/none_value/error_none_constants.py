
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Operations with None constants"
4: 
5: if __name__ == '__main__':
6:     # Type error
7:     a = 4 + None
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Operations with None constants')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a BinOp to a Name (line 7):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'int')
    # Getting the type of 'None' (line 7)
    None_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'None')
    # Applying the binary operator '+' (line 7)
    result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 8), '+', int_2, None_3)
    
    # Assigning a type to the variable 'a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'a', result_add_4)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
