
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "assert keyword with a non-boolean in the condition"
4: 
5: if __name__ == '__main__':
6:     a = 1
7:     assert a
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'assert keyword with a non-boolean in the condition')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 6):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
    # Assigning a type to the variable 'a' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', int_2)
    # Evaluating assert statement condition
    # Getting the type of 'a' (line 7)
    a_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'a')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
