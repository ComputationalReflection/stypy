
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "format builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType) -> <type 'str'>
7:     # (AnyType, Str) -> <type 'str'>
8: 
9: 
10:     # Call the builtin with correct parameters
11:     # No error
12:     ret = format(int)
13:     # No error
14:     ret = format(int, "")
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = format(int, list())
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'format builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to format(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'int' (line 12)
    int_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'int', False)
    # Processing the call keyword arguments (line 12)
    kwargs_4 = {}
    # Getting the type of 'format' (line 12)
    format_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'format', False)
    # Calling format(args, kwargs) (line 12)
    format_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), format_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', format_call_result_5)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to format(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'int' (line 14)
    int_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'int', False)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'str', '')
    # Processing the call keyword arguments (line 14)
    kwargs_9 = {}
    # Getting the type of 'format' (line 14)
    format_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'format', False)
    # Calling format(args, kwargs) (line 14)
    format_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), format_6, *[int_7, str_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', format_call_result_10)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to format(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'int' (line 18)
    int_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'int', False)
    
    # Call to list(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_14 = {}
    # Getting the type of 'list' (line 18)
    list_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'list', False)
    # Calling list(args, kwargs) (line 18)
    list_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 18, 22), list_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 18)
    kwargs_16 = {}
    # Getting the type of 'format' (line 18)
    format_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'format', False)
    # Calling format(args, kwargs) (line 18)
    format_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), format_11, *[int_12, list_call_result_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', format_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
