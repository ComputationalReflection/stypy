
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "open builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11: 
12:     # Call the builtin
13:     ret = open("f.py")
14: 
15:     # Type error
16:     ret.unexisting_method()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'open builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to open(...): (line 13)
    # Processing the call arguments (line 13)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'f.py')
    # Processing the call keyword arguments (line 13)
    kwargs_4 = {}
    # Getting the type of 'open' (line 13)
    open_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'open', False)
    # Calling open(args, kwargs) (line 13)
    open_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), open_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', open_call_result_5)
    
    # Call to unexisting_method(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_8 = {}
    # Getting the type of 'ret' (line 16)
    ret_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 16)
    unexisting_method_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), ret_6, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 16)
    unexisting_method_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), unexisting_method_7, *[], **kwargs_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
