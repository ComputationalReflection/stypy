
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "open builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11:     # Type error
12:     ret = open(str, str)
13:     # Type error
14:     ret = open(str)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'open builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to open(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'str' (line 12)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', False)
    # Getting the type of 'str' (line 12)
    str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', False)
    # Processing the call keyword arguments (line 12)
    kwargs_5 = {}
    # Getting the type of 'open' (line 12)
    open_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'open', False)
    # Calling open(args, kwargs) (line 12)
    open_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), open_2, *[str_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', open_call_result_6)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to open(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'str' (line 14)
    str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', False)
    # Processing the call keyword arguments (line 14)
    kwargs_9 = {}
    # Getting the type of 'open' (line 14)
    open_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'open', False)
    # Calling open(args, kwargs) (line 14)
    open_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), open_7, *[str_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', open_call_result_10)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
