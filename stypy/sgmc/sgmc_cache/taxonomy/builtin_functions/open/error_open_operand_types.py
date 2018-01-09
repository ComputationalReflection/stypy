
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "open builtin is invoked, but incorrect parameter types are passed"
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
13:     ret = open("f.py", "r")
14:     ret = open("f.py", "r", 0)
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = open(0)
19:     # Type error
20:     ret = open(0, "r")
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'open builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to open(...): (line 13)
    # Processing the call arguments (line 13)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'f.py')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'r')
    # Processing the call keyword arguments (line 13)
    kwargs_5 = {}
    # Getting the type of 'open' (line 13)
    open_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'open', False)
    # Calling open(args, kwargs) (line 13)
    open_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), open_2, *[str_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', open_call_result_6)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to open(...): (line 14)
    # Processing the call arguments (line 14)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'f.py')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'r')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_11 = {}
    # Getting the type of 'open' (line 14)
    open_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'open', False)
    # Calling open(args, kwargs) (line 14)
    open_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), open_7, *[str_8, str_9, int_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', open_call_result_12)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to open(...): (line 18)
    # Processing the call arguments (line 18)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_15 = {}
    # Getting the type of 'open' (line 18)
    open_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'open', False)
    # Calling open(args, kwargs) (line 18)
    open_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), open_13, *[int_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', open_call_result_16)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to open(...): (line 20)
    # Processing the call arguments (line 20)
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
    str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', 'r')
    # Processing the call keyword arguments (line 20)
    kwargs_20 = {}
    # Getting the type of 'open' (line 20)
    open_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'open', False)
    # Calling open(args, kwargs) (line 20)
    open_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), open_17, *[int_18, str_19], **kwargs_20)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', open_call_result_21)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
