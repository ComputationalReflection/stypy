
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Operations with None variables"
4: 
5: if __name__ == '__main__':
6:     none1 = None
7: 
8:     # Type error
9:     a = 4 * none1
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Operations with None variables')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Name to a Name (line 6):
    # Getting the type of 'None' (line 6)
    None_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'None')
    # Assigning a type to the variable 'none1' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'none1', None_2)
    
    # Assigning a BinOp to a Name (line 9):
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'int')
    # Getting the type of 'none1' (line 9)
    none1_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'none1')
    # Applying the binary operator '*' (line 9)
    result_mul_5 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 8), '*', int_3, none1_4)
    
    # Assigning a type to the variable 'a' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'a', result_mul_5)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
