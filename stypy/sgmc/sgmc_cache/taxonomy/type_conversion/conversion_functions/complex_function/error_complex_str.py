
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "A string parameter converted to complex"
3: 
4: if __name__ == '__main__':
5:     # Type error #
6:     print complex("23", 10)
7: 
8:     # Type error #
9:     print complex("not a number")
10: 
11:     print complex("3+2j")
12: 
13:     print complex(3, 2)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'A string parameter converted to complex')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Call to complex(...): (line 6)
    # Processing the call arguments (line 6)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'str', '23')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'complex' (line 6)
    complex_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 6)
    complex_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 10), complex_2, *[str_3, int_4], **kwargs_5)
    
    
    # Call to complex(...): (line 9)
    # Processing the call arguments (line 9)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'not a number')
    # Processing the call keyword arguments (line 9)
    kwargs_9 = {}
    # Getting the type of 'complex' (line 9)
    complex_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 9)
    complex_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 10), complex_7, *[str_8], **kwargs_9)
    
    
    # Call to complex(...): (line 11)
    # Processing the call arguments (line 11)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', '3+2j')
    # Processing the call keyword arguments (line 11)
    kwargs_13 = {}
    # Getting the type of 'complex' (line 11)
    complex_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 11)
    complex_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), complex_11, *[str_12], **kwargs_13)
    
    
    # Call to complex(...): (line 13)
    # Processing the call arguments (line 13)
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_18 = {}
    # Getting the type of 'complex' (line 13)
    complex_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 13)
    complex_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), complex_15, *[int_16, int_17], **kwargs_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
