
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Checking the type of the object the sliced is applied to"
3: 
4: if __name__ == '__main__':
5:     x = 3
6:     # Type error
7:     y = x[1:3]
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Checking the type of the object the sliced is applied to')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
    # Assigning a type to the variable 'x' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'x', int_2)
    
    # Assigning a Subscript to a Name (line 7):
    
    # Obtaining the type of the subscript
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
    slice_5 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 8), int_3, int_4, None)
    # Getting the type of 'x' (line 7)
    x_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 7)
    getitem___7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), x_6, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 7)
    subscript_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), getitem___7, slice_5)
    
    # Assigning a type to the variable 'y' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'y', subscript_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
