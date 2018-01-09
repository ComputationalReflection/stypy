
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "One single parameter of any type not convertible to int"
3: 
4: if __name__ == '__main__':
5:     # Type error #
6:     print int(list()) + 3
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'One single parameter of any type not convertible to int')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Call to int(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to list(...): (line 6)
    # Processing the call keyword arguments (line 6)
    kwargs_4 = {}
    # Getting the type of 'list' (line 6)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'list', False)
    # Calling list(args, kwargs) (line 6)
    list_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), list_3, *[], **kwargs_4)
    
    # Processing the call keyword arguments (line 6)
    kwargs_6 = {}
    # Getting the type of 'int' (line 6)
    int_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 10), 'int', False)
    # Calling int(args, kwargs) (line 6)
    int_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 6, 10), int_2, *[list_call_result_5], **kwargs_6)
    
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'int')
    # Applying the binary operator '+' (line 6)
    result_add_9 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 10), '+', int_call_result_7, int_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
