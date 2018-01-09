
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "A string parameter converted to float from a string"
3: 
4: if __name__ == '__main__':
5:     print float("23.4") + 3
6: 
7:     # Type error #
8:     print float("not a float") + 3
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'A string parameter converted to float from a string')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Call to float(...): (line 5)
    # Processing the call arguments (line 5)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', '23.4')
    # Processing the call keyword arguments (line 5)
    kwargs_4 = {}
    # Getting the type of 'float' (line 5)
    float_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'float', False)
    # Calling float(args, kwargs) (line 5)
    float_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 5, 10), float_2, *[str_3], **kwargs_4)
    
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
    # Applying the binary operator '+' (line 5)
    result_add_7 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 10), '+', float_call_result_5, int_6)
    
    
    # Call to float(...): (line 8)
    # Processing the call arguments (line 8)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 16), 'str', 'not a float')
    # Processing the call keyword arguments (line 8)
    kwargs_10 = {}
    # Getting the type of 'float' (line 8)
    float_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'float', False)
    # Calling float(args, kwargs) (line 8)
    float_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), float_8, *[str_9], **kwargs_10)
    
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'int')
    # Applying the binary operator '+' (line 8)
    result_add_13 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 10), '+', float_call_result_11, int_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
