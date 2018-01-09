
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "file builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11: 
12:     # Call the builtin with correct parameters
13:     # No error
14:     ret = file("f.py", "r", 0)
15: 
16:     # Call the builtin with incorrect types of parameters
17: 
18:     # Type error
19:     ret = file(3)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'file builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to file(...): (line 14)
    # Processing the call arguments (line 14)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'f.py')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'r')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_6 = {}
    # Getting the type of 'file' (line 14)
    file_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'file', False)
    # Calling file(args, kwargs) (line 14)
    file_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), file_2, *[str_3, str_4, int_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', file_call_result_7)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to file(...): (line 19)
    # Processing the call arguments (line 19)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_10 = {}
    # Getting the type of 'file' (line 19)
    file_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'file', False)
    # Calling file(args, kwargs) (line 19)
    file_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), file_8, *[int_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', file_call_result_11)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
