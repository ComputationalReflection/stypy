
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "A string parameter converted to int with base"
3: 
4: if __name__ == '__main__':
5:     print int("23", 10) + 3
6: 
7:     # Type error #
8:     print int("not a number", 10) + 3
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'A string parameter converted to int with base')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Call to int(...): (line 5)
    # Processing the call arguments (line 5)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'str', '23')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
    # Processing the call keyword arguments (line 5)
    kwargs_5 = {}
    # Getting the type of 'int' (line 5)
    int_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'int', False)
    # Calling int(args, kwargs) (line 5)
    int_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 5, 10), int_2, *[str_3, int_4], **kwargs_5)
    
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
    # Applying the binary operator '+' (line 5)
    result_add_8 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 10), '+', int_call_result_6, int_7)
    
    
    # Call to int(...): (line 8)
    # Processing the call arguments (line 8)
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'str', 'not a number')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'int')
    # Processing the call keyword arguments (line 8)
    kwargs_12 = {}
    # Getting the type of 'int' (line 8)
    int_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'int', False)
    # Calling int(args, kwargs) (line 8)
    int_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), int_9, *[str_10, int_11], **kwargs_12)
    
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 36), 'int')
    # Applying the binary operator '+' (line 8)
    result_add_15 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 10), '+', int_call_result_13, int_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
