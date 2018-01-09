
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Checking the return type of __getslice__"
3: 
4: if __name__ == '__main__':
5:     l = range(5)
6: 
7:     sl = l.__getslice__(0, 2)
8: 
9:     # Type error
10:     print sl + 3
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Checking the return type of __getslice__')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 5):
    
    # Call to range(...): (line 5)
    # Processing the call arguments (line 5)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
    # Processing the call keyword arguments (line 5)
    kwargs_4 = {}
    # Getting the type of 'range' (line 5)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'range', False)
    # Calling range(args, kwargs) (line 5)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'l' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'l', range_call_result_5)
    
    # Assigning a Call to a Name (line 7):
    
    # Call to __getslice__(...): (line 7)
    # Processing the call arguments (line 7)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'int')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_10 = {}
    # Getting the type of 'l' (line 7)
    l_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'l', False)
    # Obtaining the member '__getslice__' of a type (line 7)
    getslice___7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), l_6, '__getslice__')
    # Calling __getslice__(args, kwargs) (line 7)
    getslice___call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), getslice___7, *[int_8, int_9], **kwargs_10)
    
    # Assigning a type to the variable 'sl' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'sl', getslice___call_result_11)
    # Getting the type of 'sl' (line 10)
    sl_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'sl')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
    # Applying the binary operator '+' (line 10)
    result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '+', sl_12, int_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
